#!/usr/bin/env python3
"""
Unified script to generate figures from research data.
Generates:
1. Best model performance by dataset (accuracy)
2. First response similarity by dataset (one boxplot per platform)
3. Cosine similarity by model and platform (3 rows × N models)
4. ML explainability heatmap
5. Aggregated feature frequency by model
6. Raw feature bias: AI vs human distributions (top 5 features per platform)
7. Feature bias as AI − human median (top 5 features per platform)
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from matplotlib.patches import Patch, Rectangle
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

def _save_stats(df, groupby_cols, value_cols, output_path):
    """Save summary statistics (count, mean, std, median, Q1, Q3) for a figure's data."""
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    agg = df.groupby(groupby_cols)[value_cols].describe(percentiles=[.25, .5, .75])
    agg.columns = ['_'.join(c).strip('_') for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(output_path, index=False)
    print(f"  Stats saved to: {output_path}")

# Import custom utilities
from plotting_utils import (
    MODEL_PALETTE, DATASET_PALETTE, parse_filename, make_label, 
    get_ordered_models, format_dataset_name, with_plot_style,
    filter_for_baseline_persona, normalize_dataset_name, get_dataset_color
)

HUMAN_COLOR = '#888888'
HUMAN_LABEL = 'Human'

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
    
    # Load data directly here
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
                model, ft, context, style, persona = parse_filename(str(filepath))
                if model is None or "random_validation" not in str(filepath):
                    continue
                
                # Load JSON
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Extract confusion matrices (clean JSON format)
                confusion_matrices = []
                if isinstance(data, list):
                    confusion_matrices = [item['confusion_matrix'] for item in data if 'confusion_matrix' in item]
                elif 'confusion_matrix' in data:
                    confusion_matrices = [data['confusion_matrix']]
                
                if len(confusion_matrices) < 2:  # Need at least 2 for std
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
                    "dataset": normalize_dataset_name(dataset),  # Normalize dataset name
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
    
    # Group by model and dataset, select the row with max accuracy
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    best_performances = df.loc[df.groupby(['model', 'dataset'])[mean_col].idxmax()].copy()
    
    # Use normalized dataset names
    datasets = NORMALIZED_DATASETS
    
    # Get model order
    model_order = get_ordered_models(best_performances['model'].unique())
    
    with with_plot_style(PRESENTATION_MODE):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set up bar positions
        n_models = len(model_order)
        n_datasets = len(datasets)
        bar_width = 0.25
        x_positions = np.arange(n_models)
        
        # Plot bars for each dataset
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
            
            # Get dataset color and label
            dataset_color = get_dataset_color(dataset)
            dataset_label = format_dataset_name(dataset)
            
            # Plot bars with error bars
            ax.bar(x_positions + i * bar_width, dataset_data, bar_width, 
                   label=dataset_label, color=dataset_color, alpha=0.8, 
                   edgecolor='white', linewidth=0.5,
                   yerr=dataset_stds, capsize=3, error_kw={'linewidth': 1.5})
        
        # Customize plot
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=20)
        ax.tick_params(axis='y', labelsize=16)
        
        # Set x-axis
        ax.set_xticks(x_positions + bar_width * (n_datasets - 1) / 2)
        ax.set_xticklabels(model_order, rotation=45, ha='right')
        
        # Color x-tick labels with model colors
        for tick, model in zip(ax.get_xticklabels(), model_order):
            tick.set_color(MODEL_PALETTE.get(model, '#000000'))
            tick.set_fontsize(20)
        
        # Add legend
        ax.legend(loc='lower left', fontsize=20)
        
        # Set y-axis limits
        ax.set_ylim(0.5, 1.0)
        
        # Add grid
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
        
        # Find optimal response files
        response_files = list(dataset_path.rglob("*_optimal_response.json"))
        filtered_files = [f for f in response_files if '__noft__ctx0__style0__optimal_response.json' in str(f)]
        
        print(f"  Found {len(filtered_files)} matching files")
        
        for file_path in filtered_files:
            try:
                # Parse filename to get model
                model = os.path.basename(str(file_path)).split('__')[0]
                
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Handle both list and dict formats
                entries = data if isinstance(data, list) else [data]
                
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    
                    # Look for responses
                    for key in ['all_valid_responses', 'valid_responses', 'responses']:
                        if key in entry and entry[key] and len(entry[key]) > 0:
                            first_response = entry[key][0]
                            if isinstance(first_response, dict):
                                similarity = first_response.get("cosine_similarity") or first_response.get("similarity")
                                if similarity is not None:
                                    rows.append({
                                        "dataset": normalize_dataset_name(dataset),  # Normalize
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
    
    # Use normalized dataset names
    datasets = NORMALIZED_DATASETS
    
    with with_plot_style(PRESENTATION_MODE):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for boxplots
        all_data = [df[df['dataset'] == d]['similarity'].values if len(df[df['dataset'] == d]) > 0 else [] 
                    for d in datasets]
        all_colors = [get_dataset_color(d) for d in datasets]
        
        # Create boxplots
        bp = ax.boxplot(all_data, positions=range(len(datasets)), widths=0.6, patch_artist=True,
                       showfliers=True, medianprops=dict(color='red', linewidth=2),
                       flierprops=dict(marker='o', markersize=3, alpha=0.4))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Color whiskers and caps
        for i, color in enumerate(all_colors):
            for j in [i*2, i*2+1]:
                bp['whiskers'][j].set_color(color)
                bp['caps'][j].set_color(color)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Set labels
        dataset_labels = [format_dataset_name(d) for d in datasets]
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(dataset_labels, rotation=0, ha='center')
        ax.tick_params(axis='y', labelsize=16)
        
        # Color x-tick labels
        for tick, dataset in zip(ax.get_xticklabels(), datasets):
            tick.set_color(get_dataset_color(dataset))
            tick.set_fontsize(20)
        
        ax.set_ylim([-1, 1])
        ax.set_ylabel("Cosine Similarity", fontsize=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        stats_path = os.path.join(OUTPUT_DIR, f'SOTA_cosine_similarity_stats.csv')
        _save_stats(df, ['dataset'], 'similarity', stats_path)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'SOTA_cosine_similarity.{SAVE_FORMAT}')
        plt.savefig(output_path, dpi=600, bbox_inches="tight", transparent=PRESENTATION_MODE)
        plt.close()
        print(f"  Saved to: {output_path}")

# === Figure 3: Cosine Similarity by Model and Platform ===
def generate_cosine_similarity_by_model():
    """
    3-row figure (one per platform). Each row shows one boxplot per LLM,
    colored by model name. A single shared x-axis label appears at the bottom.
    """
    print("\n=== Generating Figure 3: Cosine Similarity by Model and Platform ===")

    rows = []

    for dataset in DATASETS:
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            print(f"Warning: {dataset_path} does not exist, skipping...")
            continue

        print(f"Processing {dataset}...")

        response_files = list(dataset_path.rglob("*_optimal_response.json"))
        filtered_files = [f for f in response_files
                          if '__noft__ctx0__style0__optimal_response.json' in str(f)]

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
                        if key in entry and entry[key]:
                            first = entry[key][0]
                            if isinstance(first, dict):
                                sim = first.get("cosine_similarity") or first.get("similarity")
                                if sim is not None:
                                    rows.append({
                                        "dataset": normalize_dataset_name(dataset),
                                        "model": model,
                                        "similarity": sim,
                                    })
                            break

            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")

    if not rows:
        print("No data found for cosine similarity by model plot!")
        return

    df = pd.DataFrame(rows)
    datasets = NORMALIZED_DATASETS
    model_order = get_ordered_models(df['model'].unique())

    with with_plot_style(PRESENTATION_MODE):
        fig, axes = plt.subplots(
            len(datasets), 1,
            figsize=(14, 5 * len(datasets)),
            sharex=True,
        )
        if len(datasets) == 1:
            axes = [axes]

        for row_idx, (ax, dataset) in enumerate(zip(axes, datasets)):
            plot_data = [
                df[(df['dataset'] == dataset) & (df['model'] == m)]['similarity'].values
                for m in model_order
            ]
            colors = [MODEL_PALETTE.get(m, '#888888') for m in model_order]

            bp = ax.boxplot(
                plot_data,
                positions=range(len(model_order)),
                widths=0.55,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markersize=3, alpha=0.4),
            )

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

            for i, color in enumerate(colors):
                for j in [i * 2, i * 2 + 1]:
                    bp['whiskers'][j].set_color(color)
                    bp['caps'][j].set_color(color)

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_ylim([-1, 1])
            ax.set_ylabel("Cosine Similarity", fontsize=14)
            ax.tick_params(axis='y', labelsize=13)
            ax.grid(True, alpha=0.3, axis='y')

            # Platform label on the right
            dataset_color = get_dataset_color(dataset)
            ax.set_title(format_dataset_name(dataset), fontsize=15, loc='right',
                         color=dataset_color, pad=6)

            # x-tick labels only on the bottom row
            ax.set_xticks(range(len(model_order)))
            if row_idx == len(datasets) - 1:
                ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=13)
                for tick, model in zip(ax.get_xticklabels(), model_order):
                    tick.set_color(MODEL_PALETTE.get(model, '#000000'))
            else:
                ax.set_xticklabels([])

        stats_path = os.path.join(OUTPUT_DIR, f'SOTA_cosine_similarity_by_model_stats.csv')
        _save_stats(df, ['dataset', 'model'], 'similarity', stats_path)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'SOTA_cosine_similarity_by_model.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")


# === Figure 4: ML Explainability Heatmap ===

def generate_ml_explainability_heatmap():
    """Generate ML explainability feature importance heatmap."""
    print("\n=== Generating Figure 3: ML Explainability Heatmap ===")
    
    all_data = []
    
    for dataset in DATASETS:
        dataset_path = Path(dataset)
        
        if not dataset_path.exists():
            continue
        
        print(f"Processing {dataset}...")
        
        # Find feature correlation files
        files = glob.glob(f"{dataset_path}/**/*feature_correlation_stats.csv", recursive=True)
        filtered_files = [f for f in files if filter_for_baseline_persona(f)]
        
        print(f"  Found {len(filtered_files)} matching files")
        
        for file_path in filtered_files:
            try:
                # Parse model from filename
                filename = os.path.basename(file_path)
                config_part = filename.split('___random')[0] if '___random' in filename else filename.split('__random')[0]
                
                model, _, _, _, _ = parse_filename(config_part)
                
                if model is None:
                    continue
                
                # Read data
                df = pd.read_csv(file_path)
                df['dataset'] = normalize_dataset_name(dataset)  # Normalize
                df['model'] = model
                all_data.append(df)
                
            except Exception as e:
                print(f"  Error processing {os.path.basename(file_path)}: {e}")
    
    if not all_data:
        print("No data found for ML explainability heatmap!")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined data: {combined_df.shape[0]} rows")
    
    # Format feature names
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
    model_colors = [MODEL_PALETTE.get(m, 'gray') for m in models]
    
    # Get candidate features (top overall + top per model)
    overall_importance = combined_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    top_overall = set(overall_importance.head(10).index)
    
    top_per_model = set()
    for model in models:
        model_imp = combined_df[combined_df['model'] == model].groupby('feature')['importance'].mean().sort_values(ascending=False)
        top_per_model.update(model_imp.head(2).index)
    
    candidate_features = top_overall.union(top_per_model)
    
    # Use normalized dataset names
    datasets = NORMALIZED_DATASETS
    
    with with_plot_style(PRESENTATION_MODE):
        fig, axes = plt.subplots(1, 3, figsize=(22, 5), sharey=True)
        
        # Calculate global min/max
        all_values = []
        for dataset in datasets:
            dataset_data = combined_df[combined_df['dataset'] == dataset]
            for model in models:
                model_data = dataset_data[dataset_data['model'] == model]
                all_values.extend(model_data.groupby('feature')['importance'].mean().values)
        
        vmin, vmax = min(all_values) if all_values else 0, max(all_values) if all_values else 1
        
        from matplotlib.colors import PowerNorm
        
        # Handle negative values
        if vmin <= 0:
            shift = abs(vmin) + 0.0001
            norm = PowerNorm(gamma=1.0, vmin=vmin+shift, vmax=vmax+shift)
            use_shift = True
        else:
            norm = PowerNorm(gamma=1.0, vmin=vmin, vmax=vmax)
            use_shift = False
        
        # Create heatmap for each dataset
        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            dataset_data = combined_df[combined_df['dataset'] == dataset]
            
            # Get top features for this dataset
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
            
            # Build importance matrix
            importance_matrix = []
            for model in models:
                model_data = dataset_data[dataset_data['model'] == model]
                model_importance = model_data.groupby('feature')['importance'].mean()
                importance_matrix.append([model_importance.get(f, 0) for f in selected_features])
            
            importance_array = np.array(importance_matrix)
            importance_shifted = importance_array + shift if use_shift else importance_array
            
            # Create heatmap
            im = ax.imshow(importance_shifted, cmap='gray_r', aspect='auto', norm=norm)
            
            # Format features
            formatted_features = [format_feature_name(f) for f in selected_features]
            
            ax.set_xticks(range(len(formatted_features)))
            ax.set_xticklabels(formatted_features, rotation=45, ha='right', fontsize=10)
            
            # Add y-axis labels only for first subplot
            if idx == 0:
                ax.set_yticks(range(len(models)))
                ax.set_yticklabels([])
                for i, (model, color) in enumerate(zip(models, model_colors)):
                    ax.text(-0.5, i, model, ha='right', va='center', color=color, fontsize=11)
            else:
                ax.set_yticks([])
            
            # Add values to cells
            for i in range(len(models)):
                for j in range(len(formatted_features)):
                    value = importance_array[i, j]
                    value_shifted = importance_shifted[i, j]
                    normalized = (value_shifted - (vmin+shift if use_shift else vmin)) / ((vmax+shift if use_shift else vmax) - (vmin+shift if use_shift else vmin))
                    text_color = 'white' if normalized > 0.5 else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                           color=text_color, fontsize=9)
            
            # Title
            dataset_name = format_dataset_name(dataset)
            ax.set_title(dataset_name, fontsize=14, pad=10, color=get_dataset_color(dataset))
        
        stats_path = os.path.join(OUTPUT_DIR, f'SOTA_ML_stats.csv')
        ml_stats = combined_df.groupby(['dataset', 'model', 'feature'])['importance'].mean().reset_index()
        ml_stats = ml_stats.rename(columns={'importance': 'mean_importance'})
        ml_stats.to_csv(stats_path, index=False)
        print(f"  Stats saved to: {stats_path}")
        # Add colorbar
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Feature Importance', fontsize=12)

        output_path = os.path.join(OUTPUT_DIR, f'SOTA_ML.{SAVE_FORMAT}')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close()
        print(f"  Saved to: {output_path}")

# === Figure 4: Aggregated Feature Frequency by Model ===
def generate_aggregated_feature_frequency():
    """Generate aggregated feature frequency by model figure."""
    print("\n=== Generating Figure 4: Aggregated Feature Frequency by Model ===")
    
    all_data = []
    
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
                        if clean_feature not in feature_model_counts:
                            feature_model_counts[clean_feature] = {}
                        if model not in feature_model_counts[clean_feature]:
                            feature_model_counts[clean_feature][model] = 0
                        feature_model_counts[clean_feature][model] += 1
                except Exception as e:
                    continue
        
        if feature_model_counts:
            # Get top 20 features
            feature_totals = {f: sum(counts.values()) for f, counts in feature_model_counts.items()}
            sorted_features = sorted(feature_totals.items(), key=lambda x: x[1], reverse=True)
            top_features = [f for f, _ in sorted_features[:20]]
            
            all_data.append({
                'features': top_features,
                'model_counts': {f: feature_model_counts[f] for f in top_features},
                'dataset': format_dataset_name(dataset)
            })
    
    if not all_data:
        print("No data found for aggregated feature frequency plot!")
        return
    
    # Get all unique models
    all_models = set()
    for data in all_data:
        for feature_counts in data['model_counts'].values():
            all_models.update(feature_counts.keys())
    
    ordered_models = get_ordered_models(list(all_models))
    total_models = len(ordered_models)
    
    with with_plot_style(PRESENTATION_MODE):
        fig, axes = plt.subplots(1, len(all_data), figsize=(18, 8), sharey=False)
        
        if len(all_data) == 1:
            axes = [axes]
        
        for i, (ax, data) in enumerate(zip(axes, all_data)):
            features = data['features']
            model_counts = data['model_counts']
            dataset_name = data['dataset']
            
            # Stacked horizontal bar with percentages
            bottom = np.zeros(len(features))
            
            for model in ordered_models:
                if model in all_models:
                    model_values = [model_counts.get(f, {}).get(model, 0) for f in features]
                    model_percentages = [(v / total_models * 100) for v in model_values]
                    color = MODEL_PALETTE.get(model, 'gray')
                    
                    ax.barh(range(len(features)), model_percentages, left=bottom, 
                           color=color, alpha=0.7, label=model)
                    bottom += model_percentages
            
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('Share (%)', fontsize=12)
            
            # Use get_dataset_color for the title color
            # Find matching normalized dataset
            dataset_color = '#000000'  # default
            for norm_ds in NORMALIZED_DATASETS:
                if format_dataset_name(norm_ds) == dataset_name:
                    dataset_color = get_dataset_color(norm_ds)
                    break
            
            ax.set_title(dataset_name, fontsize=14, pad=10, color=dataset_color)
            ax.set_xlim(0, 105)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Legend
        legend_elements = [Rectangle((0,0),1,1, facecolor=MODEL_PALETTE.get(m, 'gray'), 
                                    alpha=0.7, label=m) for m in ordered_models]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.1), 
                  ncol=min(len(ordered_models), 3), fontsize=10)
        
        empath_rows = []
        for data_item in all_data:
            for feat, model_counts in data_item['model_counts'].items():
                for mdl, cnt in model_counts.items():
                    empath_rows.append({'dataset': data_item['dataset'], 'feature': feat, 'model': mdl, 'count': cnt})
        if empath_rows:
            stats_path = os.path.join(OUTPUT_DIR, f'SOTA_empath_stats.csv')
            pd.DataFrame(empath_rows).to_csv(stats_path, index=False)
            print(f"  Stats saved to: {stats_path}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.2)

        output_path = os.path.join(OUTPUT_DIR, f'SOTA_empath.{SAVE_FORMAT}')
        plt.savefig(output_path, dpi=600, transparent=PRESENTATION_MODE, bbox_inches='tight')
        plt.close()
        print(f"  Saved to: {output_path}")

# === Main Execution ===
# === Feature bias helpers ===

def _get_top_features(dataset_path: Path, top_n: int = 5) -> list:
    """Return top-N features ranked by mean RF importance across SOTA-config models."""
    files = glob.glob(str(dataset_path / '**' / '*feature_correlation_stats.csv'), recursive=True)
    files = [f for f in files if filter_for_baseline_persona(f)]
    dfs = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath)
            if 'feature' in df.columns and 'importance' in df.columns:
                dfs.append(df[['feature', 'importance']])
        except Exception:
            continue
    if not dfs:
        return []
    combined = pd.concat(dfs, ignore_index=True)
    return (combined.groupby('feature')['importance']
                    .mean()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .index.tolist())


def _load_feature_distributions(dataset_path: Path) -> dict:
    """Load raw feature values + label (0=AI, 1=human) for each SOTA-config model."""
    label_files = glob.glob(str(dataset_path / '**' / '*_random_validation_data.csv'), recursive=True)
    label_files = [f for f in label_files if filter_for_baseline_persona(f) and '_features' not in f]
    model_data = {}
    for lpath in label_files:
        try:
            model, ft, context, style, persona = parse_filename(lpath)
            if model is None:
                continue
            feat_path = lpath.replace('_random_validation_data.csv', '_random_validation_data_features.csv')
            if not os.path.exists(feat_path):
                continue
            labels_df = pd.read_csv(lpath)
            feats_df  = pd.read_csv(feat_path)
            if len(labels_df) != len(feats_df):
                continue
            feats_df['label'] = labels_df['labels'].values
            model_data[model] = feats_df
        except Exception:
            continue
    return model_data


# === Figure 6: Raw Feature Bias (AI vs Human) ===
def generate_feature_bias(top_n: int = 5):
    """
    3 rows (platforms) × top_n columns (features).
    Each subplot: one boxplot per model (MODEL_PALETTE) + one Human boxplot (gray).
    """
    print(f"\n=== Generating Figure 6: Raw Feature Bias (AI vs Human) ===")

    platform_features = {}
    platform_data     = {}

    for dataset in DATASETS:
        pname = normalize_dataset_name(dataset)
        top_feats  = _get_top_features(Path(dataset), top_n)
        model_data = _load_feature_distributions(Path(dataset))
        if not top_feats or not model_data:
            print(f"  Skipping {pname}: missing data")
            continue
        print(f"  {pname}: top features = {top_feats}")
        platform_features[pname] = top_feats
        platform_data[pname]     = model_data

    if not platform_data:
        print("  No data found.")
        return

    pnames      = list(platform_data.keys())
    model_order = get_ordered_models({m for pd_ in platform_data.values() for m in pd_})
    x_labels    = model_order + [HUMAN_LABEL]
    x_positions = list(range(len(x_labels)))
    n_rows, n_cols = len(pnames), top_n

    with with_plot_style(PRESENTATION_MODE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4 * n_rows), sharey=False)
        if n_rows == 1: axes = axes[np.newaxis, :]
        if n_cols == 1: axes = axes[:, np.newaxis]

        for row_idx, pname in enumerate(pnames):
            pdata    = platform_data[pname]
            features = platform_features[pname]
            pcolor   = get_dataset_color(pname)
            first_df = next(iter(pdata.values()), None)

            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                if col_idx >= len(features):
                    ax.set_visible(False)
                    continue
                feat = features[col_idx]

                box_data, box_colors = [], []
                for model in model_order:
                    df = pdata.get(model)
                    vals = df.loc[df['label'] == 0, feat].dropna().values if (df is not None and feat in df.columns) else np.array([])
                    box_data.append(vals)
                    box_colors.append(MODEL_PALETTE.get(model, '#888888'))

                human_vals = first_df.loc[first_df['label'] == 1, feat].dropna().values if (first_df is not None and feat in first_df.columns) else np.array([])
                human_median = float(np.median(human_vals)) if len(human_vals) else 0.0
                box_data.append(human_vals)
                box_colors.append(HUMAN_COLOR)

                bp = ax.boxplot(box_data, positions=x_positions, widths=0.55, patch_artist=True,
                                showfliers=False, medianprops=dict(color='black', linewidth=1.5))
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color); patch.set_alpha(0.75)
                for i, color in enumerate(box_colors):
                    bp['whiskers'][i*2].set_color(color);   bp['whiskers'][i*2+1].set_color(color)
                    bp['caps'][i*2].set_color(color);       bp['caps'][i*2+1].set_color(color)

                ax.axhline(human_median, color=HUMAN_COLOR, linestyle='--', linewidth=1, alpha=0.8)
                ax.set_title(feat.replace('_', ' '), fontsize=12, pad=4)
                if col_idx == 0:
                    ax.set_ylabel(format_dataset_name(pname), fontsize=13, color=pcolor, labelpad=6)
                ax.tick_params(axis='y', labelsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xticks(x_positions)
                if row_idx == n_rows - 1:
                    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
                    for tick, lbl in zip(ax.get_xticklabels(), x_labels):
                        tick.set_color(HUMAN_COLOR if lbl == HUMAN_LABEL else MODEL_PALETTE.get(lbl, '#000000'))
                else:
                    ax.set_xticklabels([])

        legend_handles = [mpatches.Patch(facecolor=MODEL_PALETTE.get(m, '#888888'), alpha=0.75, label=m) for m in model_order]
        legend_handles.append(mpatches.Patch(facecolor=HUMAN_COLOR, alpha=0.75, label=HUMAN_LABEL))
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                   ncol=min(len(legend_handles), 5), fontsize=11, frameon=True)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18)
        output_path = os.path.join(OUTPUT_DIR, f'SOTA_feature_bias.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")


# === Figure 7: Feature Bias — AI deviation from human median ===
def generate_feature_bias_diff(top_n: int = 5):
    """
    3 rows (platforms) × top_n columns (features).
    Each value is AI_feature - human_median; dashed line at 0 = no bias.
    """
    print(f"\n=== Generating Figure 7: Feature Bias (AI − human median) ===")

    platform_features = {}
    platform_data     = {}

    for dataset in DATASETS:
        pname = normalize_dataset_name(dataset)
        top_feats  = _get_top_features(Path(dataset), top_n)
        model_data = _load_feature_distributions(Path(dataset))
        if not top_feats or not model_data:
            continue
        platform_features[pname] = top_feats
        platform_data[pname]     = model_data

    if not platform_data:
        print("  No data found.")
        return

    pnames      = list(platform_data.keys())
    model_order = get_ordered_models({m for pd_ in platform_data.values() for m in pd_})
    x_positions = list(range(len(model_order)))
    n_rows, n_cols = len(pnames), top_n

    with with_plot_style(PRESENTATION_MODE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4 * n_rows), sharey=False)
        if n_rows == 1: axes = axes[np.newaxis, :]
        if n_cols == 1: axes = axes[:, np.newaxis]

        for row_idx, pname in enumerate(pnames):
            pdata    = platform_data[pname]
            features = platform_features[pname]
            pcolor   = get_dataset_color(pname)
            first_df = next(iter(pdata.values()), None)

            human_medians = {}
            if first_df is not None:
                for feat in features:
                    if feat in first_df.columns:
                        hv = first_df.loc[first_df['label'] == 1, feat].dropna()
                        human_medians[feat] = float(hv.median()) if len(hv) else 0.0

            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                if col_idx >= len(features):
                    ax.set_visible(False)
                    continue
                feat         = features[col_idx]
                human_median = human_medians.get(feat, 0.0)

                box_data, box_colors = [], []
                for model in model_order:
                    df = pdata.get(model)
                    if df is not None and feat in df.columns:
                        diff = df.loc[df['label'] == 0, feat].dropna().values - human_median
                    else:
                        diff = np.array([])
                    box_data.append(diff)
                    box_colors.append(MODEL_PALETTE.get(model, '#888888'))

                bp = ax.boxplot(box_data, positions=x_positions, widths=0.55, patch_artist=True,
                                showfliers=False, medianprops=dict(color='black', linewidth=1.5))
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color); patch.set_alpha(0.75)
                for i, color in enumerate(box_colors):
                    bp['whiskers'][i*2].set_color(color);   bp['whiskers'][i*2+1].set_color(color)
                    bp['caps'][i*2].set_color(color);       bp['caps'][i*2+1].set_color(color)

                ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
                ax.set_title(feat.replace('_', ' '), fontsize=12, pad=4)
                if col_idx == 0:
                    ax.set_ylabel(f"{format_dataset_name(pname)}\nAI − human median",
                                  fontsize=12, color=pcolor, labelpad=6)
                ax.tick_params(axis='y', labelsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xticks(x_positions)
                if row_idx == n_rows - 1:
                    ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=10)
                    for tick, lbl in zip(ax.get_xticklabels(), model_order):
                        tick.set_color(MODEL_PALETTE.get(lbl, '#000000'))
                else:
                    ax.set_xticklabels([])

        legend_handles = [mpatches.Patch(facecolor=MODEL_PALETTE.get(m, '#888888'), alpha=0.75, label=m) for m in model_order]
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                   ncol=min(len(legend_handles), 5), fontsize=11, frameon=True)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18)
        output_path = os.path.join(OUTPUT_DIR, f'SOTA_feature_bias_diff.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")


def main():
    """Main function to generate all 4 required figures."""
    global DATASETS, NORMALIZED_DATASETS, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Generate SOTA figures from results folder.")
    parser.add_argument("results_folder", nargs="?", default=None,
                        help="Base results folder (e.g. results_2026_03_17). "
                             "Platform subfolders (bluesky, reddit, twitter) are detected automatically.")
    args = parser.parse_args()

    if args.results_folder is not None:
        base = Path(args.results_folder)
        found = sorted([str(p) for p in base.iterdir() if p.is_dir() and not p.name.endswith('_figures')])
        if not found:
            print(f"Error: no subdirectories found in {base}", file=sys.stderr)
            sys.exit(1)
        DATASETS = found
        NORMALIZED_DATASETS = [normalize_dataset_name(d) for d in DATASETS]
        OUTPUT_DIR = str(base / "SOTA_figures")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Unified Figure Generation Script")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Datasets: {DATASETS}")
    print("Configuration filter: baseline + persona only (no ft, no style, no context)")
    
    # Generate all figures
    generate_best_model_performance()
    generate_first_response_similarity()
    generate_cosine_similarity_by_model()
    generate_ml_explainability_heatmap()
    generate_aggregated_feature_frequency()
    generate_feature_bias()
    generate_feature_bias_diff()
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output files in: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
