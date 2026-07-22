#!/usr/bin/env python3
"""
Reference-configuration ("SOTA": baseline + persona, no style/context/fine-tuning)
figures that appear in the main text:
  Figure 1 - Best model performance by dataset (accuracy)
  Figure 2 - First response cosine similarity by dataset
  Figure 3 - ML explainability heatmap
  Figure 4 - feeds plot_empath_feature_heatmap.py (Empath feature significance)

For the SI-only reference-configuration figures (cosine similarity by model,
raw feature bias, feature importance by model, MDI vs. permutation importance),
see generate_SOTA_SI_figures.py.

Usage:
    python analysis/generate_SOTA_main_figures.py results_cleaned/
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.src.plotting_utils import (
    get_ordered_models, format_dataset_name, with_plot_style,
    filter_for_baseline_persona, normalize_dataset_name, get_dataset_color,
    parse_filename,
)
from analysis._sota_common import _save_stats

# === Configuration ===
DATASETS = ['results_cleaned_20260309_095640/bluesky']
OUTPUT_DIR = "results_revision/SOTA_figures"
PRESENTATION_MODE = False
SAVE_FORMAT = 'png'
NORMALIZED_DATASETS = []  # populated in main()


# === Figure 1: Best Model Performance by Dataset ===
def generate_best_model_performance(response_type="random", metric="accuracy"):
    """
    Bar plot showing the best accuracy achieved by each model across different datasets.
    Each model gets grouped bars showing performance on each dataset, with error bars.
    """
    print("\n=== Generating Figure 1: Best Model Performance by Dataset ===")

    all_results = []

    for dataset in DATASETS:
        if not os.path.exists(dataset):
            print(f"Warning: Dataset {dataset} does not exist, skipping...")
            continue

        print(f"Processing dataset: {dataset}")

        json_files = list(Path(dataset).rglob("*trainer_results.json"))
        json_files = [f for f in json_files if filter_for_baseline_persona(str(f))]
        print(f"  Found {len(json_files)} matching files")

        for filepath in json_files:
            try:
                model, ft, context, style, persona = parse_filename(str(filepath))
                if model is None or "random_validation" not in str(filepath):
                    continue

                with open(filepath, 'r') as f:
                    data = json.load(f)

                confusion_matrices = []
                if isinstance(data, list):
                    confusion_matrices = [item['confusion_matrix'] for item in data if 'confusion_matrix' in item]
                elif 'confusion_matrix' in data:
                    confusion_matrices = [data['confusion_matrix']]

                if len(confusion_matrices) < 2:  # Need at least 2 for std
                    continue

                accuracies = []
                for cm in confusion_matrices:
                    cm_array = np.array(cm)
                    total = cm_array.sum()
                    correct = cm_array[0, 0] + cm_array[1, 1]
                    accuracies.append(correct / total if total > 0 else 0)

                all_results.append({
                    "model": model,
                    "dataset": normalize_dataset_name(dataset),
                    "accuracy_mean": np.mean(accuracies),
                    "accuracy_std": np.std(accuracies, ddof=1),
                })

            except Exception as e:
                print(f"  Failed to process {filepath.name}: {e}")
                continue

    if not all_results:
        print("No data loaded!")
        return

    df = pd.DataFrame(all_results)
    print(f"Combined data: {len(df)} rows, {len(df['dataset'].unique())} datasets, {len(df['model'].unique())} models")

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    best_performances = df.loc[df.groupby(['model', 'dataset'])[mean_col].idxmax()].copy()

    datasets = NORMALIZED_DATASETS
    model_order = get_ordered_models(best_performances['model'].unique())

    with with_plot_style(PRESENTATION_MODE):
        fig, ax = plt.subplots(figsize=(14, 8))

        n_models = len(model_order)
        n_datasets = len(datasets)
        bar_width = 0.25
        x_positions = np.arange(n_models)

        for i, dataset in enumerate(datasets):
            dataset_data = []
            dataset_stds = []

            for model in model_order:
                row = best_performances[(best_performances['model'] == model) &
                                         (best_performances['dataset'] == dataset)]
                if not row.empty:
                    dataset_data.append(row[mean_col].iloc[0])
                    dataset_stds.append(row[std_col].iloc[0] if pd.notna(row[std_col].iloc[0]) else 0)
                else:
                    dataset_data.append(0)
                    dataset_stds.append(0)

            dataset_color = get_dataset_color(dataset)
            dataset_label = format_dataset_name(dataset)

            ax.bar(x_positions + i * bar_width, dataset_data, bar_width,
                   label=dataset_label, color=dataset_color, alpha=0.8,
                   edgecolor='white', linewidth=0.5,
                   yerr=dataset_stds, capsize=3, error_kw={'linewidth': 1.5})

        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=24)
        ax.tick_params(axis='y', labelsize=20)

        ax.set_xticks(x_positions + bar_width * (n_datasets - 1) / 2)
        ax.set_xticklabels(model_order, rotation=45, ha='right')

        for tick, model in zip(ax.get_xticklabels(), model_order):
            tick.set_color('black')
            tick.set_fontsize(24)

        ax.legend(loc='lower left', fontsize=24)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        stats_path = os.path.join(OUTPUT_DIR, f'SOTA_accuracy_stats.csv')
        best_performances.to_csv(stats_path, index=False)
        print(f"  Stats saved to: {stats_path}")
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'SOTA_accuracy.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")


# === Figure 2: First Response Similarity by Dataset ===
def generate_first_response_similarity():
    """Generate first response cosine similarity by dataset figure."""
    print("\n=== Generating Figure 2: First Response Similarity by Dataset ===")

    rows = []

    for dataset in DATASETS:
        dataset_path = Path(dataset)

        if not dataset_path.exists():
            print(f"Warning: {dataset_path} does not exist, skipping...")
            continue

        print(f"Processing {dataset}...")

        response_files = list(dataset_path.rglob("*_optimal_response.json"))
        filtered_files = [f for f in response_files if '__noft__ctx0__style0__optimal_response.json' in str(f)]

        print(f"  Found {len(filtered_files)} matching files")

        for file_path in filtered_files:
            try:
                model = os.path.basename(str(file_path)).split('__')[0]

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                entries = data if isinstance(data, list) else [data]

                for entry in entries:
                    if not isinstance(entry, dict):
                        continue

                    for key in ['all_valid_responses', 'valid_responses', 'responses']:
                        if key in entry and entry[key] and len(entry[key]) > 0:
                            first_response = entry[key][0]
                            if isinstance(first_response, dict):
                                similarity = first_response.get("cosine_similarity") or first_response.get("similarity")
                                if similarity is not None:
                                    rows.append({
                                        "dataset": normalize_dataset_name(dataset),
                                        "model": model,
                                        "similarity": similarity
                                    })
                            break

            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")

    print(f"Total similarity values collected: {len(rows)}")

    if not rows:
        print("No data found for similarity plot!")
        return

    df = pd.DataFrame(rows)
    datasets = NORMALIZED_DATASETS

    with with_plot_style(PRESENTATION_MODE):
        fig, ax = plt.subplots(figsize=(14, 8))

        all_data = [df[df['dataset'] == d]['similarity'].values if len(df[df['dataset'] == d]) > 0 else []
                    for d in datasets]
        all_colors = [get_dataset_color(d) for d in datasets]

        bp = ax.boxplot(all_data, positions=range(len(datasets)), widths=0.6, patch_artist=True,
                         showfliers=True, medianprops=dict(color='red', linewidth=2),
                         flierprops=dict(marker='o', markersize=3, alpha=0.4))

        for patch, color in zip(bp['boxes'], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, color in enumerate(all_colors):
            for j in [i * 2, i * 2 + 1]:
                bp['whiskers'][j].set_color(color)
                bp['caps'][j].set_color(color)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        dataset_labels = [format_dataset_name(d) for d in datasets]
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(dataset_labels, rotation=0, ha='center')
        ax.tick_params(axis='y', labelsize=20)

        for tick, dataset in zip(ax.get_xticklabels(), datasets):
            tick.set_color(get_dataset_color(dataset))
            tick.set_fontsize(24)

        ax.set_ylim([-1, 1])
        ax.set_ylabel("Cosine Similarity", fontsize=24)
        ax.grid(True, alpha=0.3, axis='y')

        stats_path = os.path.join(OUTPUT_DIR, f'SOTA_cosine_similarity_stats.csv')
        _save_stats(df, ['dataset'], 'similarity', stats_path)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'SOTA_cosine_similarity.{SAVE_FORMAT}')
        plt.savefig(output_path, dpi=600, bbox_inches="tight", transparent=PRESENTATION_MODE)
        plt.close()
        print(f"  Saved to: {output_path}")


# === Figure 3: ML Explainability Heatmap ===
def generate_ml_explainability_heatmap():
    """Generate ML explainability feature importance heatmap."""
    print("\n=== Generating Figure 3: ML Explainability Heatmap ===")

    all_data = []

    for dataset in DATASETS:
        dataset_path = Path(dataset)

        if not dataset_path.exists():
            continue

        print(f"Processing {dataset}...")

        files = glob.glob(f"{dataset_path}/**/*feature_correlation_stats.csv", recursive=True)
        filtered_files = [f for f in files if filter_for_baseline_persona(f)]

        print(f"  Found {len(filtered_files)} matching files")

        for file_path in filtered_files:
            try:
                filename = os.path.basename(file_path)
                config_part = filename.split('___random')[0] if '___random' in filename else filename.split('__random')[0]

                model, _, _, _, _ = parse_filename(config_part)

                if model is None:
                    continue

                df = pd.read_csv(file_path)
                df['dataset'] = normalize_dataset_name(dataset)
                df['model'] = model
                all_data.append(df)

            except Exception as e:
                print(f"  Error processing {os.path.basename(file_path)}: {e}")

    if not all_data:
        print("No data found for ML explainability heatmap!")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined data: {combined_df.shape[0]} rows")

    def format_feature_name(name):
        special = {
            'avg_word_length': 'Average Word Length',
            'type_token_ratio': 'Type-Token Ratio',
            'has_question_mark': 'Has Question Mark',
            'has_exclamation_mark': 'Has Exclamation Mark',
            'abstract_concrete_ratio': 'Abstract-Concrete Ratio'
        }
        return special.get(name, name.replace('_', ' ').title())

    models = get_ordered_models(combined_df['model'].unique())

    overall_importance = combined_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    top_overall = set(overall_importance.head(10).index)

    top_per_model = set()
    for model in models:
        model_imp = combined_df[combined_df['model'] == model].groupby('feature')['importance'].mean().sort_values(ascending=False)
        top_per_model.update(model_imp.head(2).index)

    candidate_features = top_overall.union(top_per_model)

    datasets = NORMALIZED_DATASETS

    with with_plot_style(PRESENTATION_MODE):
        fig, axes = plt.subplots(1, 3, figsize=(22, 5), sharey=True)

        all_values = []
        for dataset in datasets:
            dataset_data = combined_df[combined_df['dataset'] == dataset]
            for model in models:
                model_data = dataset_data[dataset_data['model'] == model]
                all_values.extend(model_data.groupby('feature')['importance'].mean().values)

        vmin, vmax = min(all_values) if all_values else 0, max(all_values) if all_values else 1

        from matplotlib.colors import PowerNorm

        if vmin <= 0:
            shift = abs(vmin) + 0.0001
            norm = PowerNorm(gamma=1.0, vmin=vmin + shift, vmax=vmax + shift)
            use_shift = True
        else:
            norm = PowerNorm(gamma=1.0, vmin=vmin, vmax=vmax)
            use_shift = False

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            dataset_data = combined_df[combined_df['dataset'] == dataset]

            dataset_importance = dataset_data.groupby('feature')['importance'].mean().sort_values(ascending=False)
            dataset_top = set(dataset_importance.head(10).index)

            dataset_top_per_model = set()
            for model in models:
                model_data = dataset_data[dataset_data['model'] == model]
                model_imp = model_data.groupby('feature')['importance'].mean().sort_values(ascending=False)
                dataset_top_per_model.update(model_imp.head(2).index)

            selected_features = sorted(
                dataset_top.union(dataset_top_per_model).intersection(candidate_features),
                key=lambda x: dataset_importance.get(x, 0),
                reverse=True
            )[:10]

            importance_matrix = []
            for model in models:
                model_data = dataset_data[dataset_data['model'] == model]
                model_importance = model_data.groupby('feature')['importance'].mean()
                importance_matrix.append([model_importance.get(f, 0) for f in selected_features])

            importance_array = np.array(importance_matrix)
            importance_shifted = importance_array + shift if use_shift else importance_array

            im = ax.imshow(importance_shifted, cmap='Blues', aspect='auto', norm=norm)

            formatted_features = [format_feature_name(f) for f in selected_features]

            ax.set_xticks(range(len(formatted_features)))
            ax.set_xticklabels(formatted_features, rotation=45, ha='right', fontsize=14)

            if idx == 0:
                ax.set_yticks(range(len(models)))
                ax.set_yticklabels([])
                for i, model in enumerate(models):
                    ax.text(-0.5, i, model, ha='right', va='center', color='black', fontsize=14)
            else:
                ax.set_yticks([])

            for i in range(len(models)):
                for j in range(len(formatted_features)):
                    value = importance_array[i, j]
                    value_shifted = importance_shifted[i, j]
                    normalized = (value_shifted - (vmin + shift if use_shift else vmin)) / ((vmax + shift if use_shift else vmax) - (vmin + shift if use_shift else vmin))
                    text_color = 'white' if normalized > 0.5 else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                            color=text_color, fontsize=9)

            dataset_name = format_dataset_name(dataset)
            ax.set_title(dataset_name, fontsize=15, pad=10, color=get_dataset_color(dataset))

        stats_path = os.path.join(OUTPUT_DIR, f'SOTA_ML_stats.csv')
        ml_stats = combined_df.groupby(['dataset', 'model', 'feature'])['importance'].mean().reset_index()
        ml_stats = ml_stats.rename(columns={'importance': 'mean_importance'})
        ml_stats.to_csv(stats_path, index=False)
        print(f"  Stats saved to: {stats_path}")
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Feature Importance', fontsize=14)

        output_path = os.path.join(OUTPUT_DIR, f'SOTA_ML.{SAVE_FORMAT}')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close()
        print(f"  Saved to: {output_path}")


# === Figure 4 (data prep): Aggregated Empath Feature Significance by Model ===
def prepare_empath_feature_stats():
    """
    Walk reference-configuration (baseline + persona) results and, per platform,
    count how many times each Empath feature reaches significance
    (adjusted p < 0.05) for each model, keeping the top-20 features per platform.

    Writes SOTA_empath_stats.csv, consumed by plot_empath_feature_heatmap.py to
    produce Figure 4. This function only prepares data -- the actual figure is
    the features x models heatmap (see plot_empath_feature_heatmap.py), which
    replaced this analysis's original stacked-bar rendering.
    """
    print("\n=== Preparing Figure 4 data: Empath Feature Significance by Model ===")

    empath_rows = []

    for dataset in DATASETS:
        dataset_path = Path(dataset)

        if not dataset_path.exists():
            continue

        print(f"Processing {dataset}...")

        feature_model_counts = {}

        for root, _, files in os.walk(dataset_path):
            for file in files:
                if not file.endswith("empath_significant_features.csv"):
                    continue

                filepath = os.path.join(root, file)
                if not filter_for_baseline_persona(filepath):
                    continue

                try:
                    model, _, _, _, _ = parse_filename(file)
                    df = pd.read_csv(filepath)
                    sig_features = df[df["adjusted_p_value"] < 0.05]["feature"].tolist()

                    for feature in sig_features:
                        clean_feature = feature.replace("_", " ")
                        feature_model_counts.setdefault(clean_feature, {})
                        feature_model_counts[clean_feature][model] = feature_model_counts[clean_feature].get(model, 0) + 1
                except Exception:
                    continue

        if not feature_model_counts:
            continue

        feature_totals = {f: sum(counts.values()) for f, counts in feature_model_counts.items()}
        top_features = [f for f, _ in sorted(feature_totals.items(), key=lambda x: x[1], reverse=True)[:20]]

        dataset_label = format_dataset_name(dataset)
        for feat in top_features:
            for mdl, cnt in feature_model_counts[feat].items():
                empath_rows.append({'dataset': dataset_label, 'feature': feat, 'model': mdl, 'count': cnt})

    if not empath_rows:
        print("  No data found for Empath feature stats!")
        return

    stats_path = os.path.join(OUTPUT_DIR, 'SOTA_empath_stats.csv')
    pd.DataFrame(empath_rows).to_csv(stats_path, index=False)
    print(f"  Stats saved to: {stats_path}")


def main():
    global DATASETS, NORMALIZED_DATASETS, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Generate main-text reference-configuration (SOTA) figures.")
    parser.add_argument("results_folder", nargs="?", default=None,
                         help="Base results folder (e.g. results_2026_03_17). "
                              "Platform subfolders (bluesky, reddit, twitter) are detected automatically.")
    args = parser.parse_args()

    if args.results_folder is not None:
        base = Path(args.results_folder)
        EXCLUDE_DIRS = {'cosine_baselines', 'configuration_optimization_figures', 'figures'}
        PLATFORM_ORDER = ['bluesky', 'twitter', 'reddit']
        all_dirs = {p.name: str(p) for p in base.iterdir() if p.is_dir() and not p.name.endswith('_figures') and p.name not in EXCLUDE_DIRS}
        found = [all_dirs[k] for k in PLATFORM_ORDER if k in all_dirs] + \
                [v for k, v in sorted(all_dirs.items()) if k not in PLATFORM_ORDER]
        if not found:
            print(f"Error: no subdirectories found in {base}", file=sys.stderr)
            sys.exit(1)
        DATASETS = found
        NORMALIZED_DATASETS = [normalize_dataset_name(d) for d in DATASETS]
        OUTPUT_DIR = str(base / "figures" / "main_figures")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SOTA Main-Text Figure Generation")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Datasets: {DATASETS}")
    print("Configuration filter: baseline + persona only (no ft, no style, no context)")

    generate_best_model_performance()
    generate_first_response_similarity()
    generate_ml_explainability_heatmap()
    prepare_empath_feature_stats()

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output files in: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
