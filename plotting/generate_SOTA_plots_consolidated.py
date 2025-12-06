#!/usr/bin/env python3
"""
Unified script to generate 4 specific figures from research data (CONSOLIDATED FORMAT VERSION).
Generates:
1. Best model performance by dataset (accuracy)
2. First response similarity by dataset
3. ML explainability heatmap
4. Aggregated feature frequency by model

This version reads from postprocessing_consolidated/ and results_consolidated/
directories instead of the old scattered file format.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from matplotlib.patches import Patch, Rectangle
import warnings
warnings.filterwarnings('ignore')

# Import custom utilities
from plotting_utils import (
    MODEL_PALETTE, DATASET_PALETTE, parse_filename, make_label,
    get_ordered_models, format_dataset_name, with_plot_style,
    filter_for_baseline_persona, normalize_dataset_name, get_dataset_color
)

# === Configuration ===
PRESENTATION_MODE = False
SAVE_FORMAT = 'png'


# === Figure 1: Best Model Performance by Dataset ===
def generate_best_model_performance(DATASETS, OUTPUT_DIR, response_type="random", metric="accuracy"):
    """
    Bar plot showing the best accuracy achieved by each model across different datasets.
    Each model gets grouped bars showing performance on each dataset, with error bars.
    """
    print("\n=== Generating Figure 1: Best Model Performance by Dataset ===")
    
    # Load data directly here
    all_results = []
    # Normalize dataset names for consistent matching
    NORMALIZED_DATASETS = [normalize_dataset_name(d) for d in DATASETS]

    for dataset in DATASETS:
        # Use postprocessing_consolidated for validation results
        # Transform: results/results_bluesky -> postprocessing_consolidated/bluesky
        consolidated_path = dataset.replace('results/results_', 'postprocessing_consolidated/')

        if not os.path.exists(consolidated_path):
            print(f"Warning: {consolidated_path} does not exist, skipping...")
            continue

        print(f"Processing dataset: {dataset}")
        dataset_name = normalize_dataset_name(dataset)

        # Find all validation_results.json files in consolidated format
        all_json_files = list(Path(consolidated_path).rglob("validation_results.json"))
        # Filter for baseline+persona only (noft, ctx0, style0, no_OPPU, not no_persona)
        json_files = []
        for f in all_json_files:
            config_dir = f.parent.name
            if '__noft__ctx0__style0__no_OPPU' in config_dir and '__no_persona' not in config_dir:
                json_files.append(f)
        print(f"  Found {len(json_files)} matching validation files (out of {len(all_json_files)} total)")

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

                # Extract accuracies from random validation (baseline+persona)
                if 'random' not in data.get('results', {}):
                    continue

                random_results = data['results']['random']
                trainer_results = random_results.get('trainer_results', [])

                if len(trainer_results) < 2:  # Need at least 2 for std
                    continue

                # Extract accuracies from individual runs
                accuracies = [run.get('accuracy', 0) for run in trainer_results
                             if 'accuracy' in run]

                if not accuracies:
                    continue

                all_results.append({
                    "model": model,
                    "dataset": dataset_name,
                    "accuracy_mean": np.mean(accuracies),
                    "accuracy_std": np.std(accuracies, ddof=1),
                })

            except Exception as e:
                print(f"  Failed to process {filepath.parent.name}: {e}")
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
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'SOTA_accuracy.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")

# === Figure 2: First Response Similarity by Dataset ===
def generate_first_response_similarity(DATASETS, OUTPUT_DIR):
    """Generate first response cosine similarity by dataset figure."""
    print("\n=== Generating Figure 2: First Response Similarity by Dataset ===")
    
    rows = []
    # Normalize dataset names for consistent matching
    NORMALIZED_DATASETS = [normalize_dataset_name(d) for d in DATASETS]

    for dataset in DATASETS:
        # Use results_consolidated for response data
        # Transform: results/results_bluesky -> results_consolidated/bluesky
        consolidated_path = dataset.replace('results/results_', 'results_consolidated/')

        if not os.path.exists(consolidated_path):
            print(f"Warning: {consolidated_path} does not exist, skipping...")
            continue

        print(f"Processing {dataset}...")
        dataset_name = normalize_dataset_name(dataset)

        # Find response files in consolidated format
        response_dir = Path(consolidated_path) / "responses"
        if not response_dir.exists():
            continue

        response_files = list(response_dir.glob("*.json"))
        # Filter for baseline+persona only (noft, ctx0, style0, no_OPPU)
        filtered_files = [f for f in response_files if '__noft__ctx0__style0__no_OPPU' in str(f)]

        print(f"  Found {len(filtered_files)} matching files")

        for file_path in filtered_files:
            try:
                # Parse filename to get model
                model, ft, context, style, oppu, persona = parse_filename(file_path.stem)

                if model is None:
                    continue

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract first response from each entry
                for response_entry in data.get('responses', []):
                    candidates = response_entry.get('candidates', [])

                    if not candidates or len(candidates) == 0:
                        continue

                    # Get first candidate's cosine similarity
                    first_candidate = candidates[0]
                    similarity = first_candidate.get('cosine_similarity')

                    if similarity is not None:
                        rows.append({
                            "dataset": dataset_name,
                            "model": model,
                            "similarity": similarity
                        })

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
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'SOTA_cosine_similarity.{SAVE_FORMAT}')
        plt.savefig(output_path, dpi=600, bbox_inches="tight", transparent=PRESENTATION_MODE)
        plt.close()
        print(f"  Saved to: {output_path}")

# === Figure 3: ML Explainability Heatmap ===
def generate_ml_explainability_heatmap(DATASETS, OUTPUT_DIR):
    """Generate ML explainability feature importance heatmap."""
    print("\n=== Generating Figure 3: ML Explainability Heatmap ===")
    
    all_data = []
    # Normalize dataset names for consistent matching
    NORMALIZED_DATASETS = [normalize_dataset_name(d) for d in DATASETS]

    for dataset in DATASETS:
        # Use postprocessing_consolidated for statistics
        # Transform: results/results_bluesky -> postprocessing_consolidated/bluesky
        consolidated_path = dataset.replace('results/results_', 'postprocessing_consolidated/')

        if not os.path.exists(consolidated_path):
            continue

        print(f"Processing {dataset}...")
        dataset_name = normalize_dataset_name(dataset)

        # Find statistics.json files in consolidated format
        all_stats_files = list(Path(consolidated_path).rglob("statistics.json"))
        # Filter for baseline+persona only (noft, ctx0, style0, no_OPPU, not no_persona)
        filtered_files = []
        for f in all_stats_files:
            config_dir = f.parent.name
            if '__noft__ctx0__style0__no_OPPU' in config_dir and '__no_persona' not in config_dir:
                filtered_files.append(f)

        print(f"  Found {len(filtered_files)} matching statistics files (out of {len(all_stats_files)} total)")

        for file_path in filtered_files:
            try:
                # Parse config from directory name
                config_dir = file_path.parent.name
                model, ft, context, style, oppu, persona = parse_filename(config_dir)

                if model is None:
                    continue

                # Load statistics
                with open(file_path, 'r') as f:
                    stats = json.load(f)

                # Extract feature correlation stats for random method
                if 'random' not in stats.get('statistics', {}):
                    continue

                random_stats = stats['statistics']['random']
                correlation_stats = random_stats.get('correlation_stats', [])

                # Convert to DataFrame format
                for feat_data in correlation_stats:
                    if isinstance(feat_data, dict):
                        all_data.append({
                            'dataset': dataset_name,
                            'model': model,
                            'feature': feat_data.get('feature', ''),
                            'importance': feat_data.get('importance', 0),
                            'correlation_sign': feat_data.get('correlation_sign', '')
                        })

            except Exception as e:
                print(f"  Error processing {file_path.parent.name}: {e}")

    # Convert list of dicts to DataFrame
    if all_data:
        combined_df = pd.DataFrame(all_data)
        all_data = [combined_df]  # Put back in list for compatibility with concat below
    else:
        all_data = []
    
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
def generate_aggregated_feature_frequency(DATASETS, OUTPUT_DIR):
    """Generate aggregated feature frequency by model figure."""
    print("\n=== Generating Figure 4: Aggregated Feature Frequency by Model ===")
    
    all_data = []
    # Normalize dataset names for consistent matching
    NORMALIZED_DATASETS = [normalize_dataset_name(d) for d in DATASETS]

    for dataset in DATASETS:
        # Use postprocessing_consolidated for statistics
        # Transform: results/results_bluesky -> postprocessing_consolidated/bluesky
        consolidated_path = dataset.replace('results/results_', 'postprocessing_consolidated/')

        if not os.path.exists(consolidated_path):
            continue

        print(f"Processing {dataset}...")
        dataset_name = normalize_dataset_name(dataset)

        feature_model_counts = {}

        # Find statistics.json files in consolidated format
        all_stats_files = list(Path(consolidated_path).rglob("statistics.json"))
        # Filter for baseline+persona only (noft, ctx0, style0, no_OPPU, not no_persona)
        filtered_files = []
        for f in all_stats_files:
            config_dir = f.parent.name
            if '__noft__ctx0__style0__no_OPPU' in config_dir and '__no_persona' not in config_dir:
                filtered_files.append(f)

        for file_path in filtered_files:
            try:
                # Parse config from directory name
                config_dir = file_path.parent.name
                model, ft, context, style, oppu, persona = parse_filename(config_dir)

                if model is None:
                    continue

                # Load statistics
                with open(file_path, 'r') as f:
                    stats = json.load(f)

                # Extract empath significant features for random method
                if 'random' not in stats.get('statistics', {}):
                    continue

                random_stats = stats['statistics']['random']
                empath_features = random_stats.get('empath_significant', [])

                # Extract significant features (p < 0.05)
                for feat_data in empath_features:
                    if isinstance(feat_data, dict):
                        feature = feat_data.get('feature', '')
                        p_value = feat_data.get('adjusted_p_value',
                                               feat_data.get('p_value', 1.0))

                        if p_value < 0.05:
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
                'dataset': format_dataset_name(dataset_name)
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
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.2)
        
        output_path = os.path.join(OUTPUT_DIR, f'SOTA_empath.{SAVE_FORMAT}')
        plt.savefig(output_path, dpi=600, transparent=PRESENTATION_MODE, bbox_inches='tight')
        plt.close()
        print(f"  Saved to: {output_path}")

# === Main Execution ===
def main(folder="results"):
    """
    Main function to generate all 4 required figures.
    
    Args:
        updated_folder (str): Base folder path for results. Defaults to "updated_results".
                              OUTPUT_DIR will be created as {updated_folder}/SOTA_plots
    """
    # Construct dataset paths from the updated_folder argument
    DATASETS = [
        f'{folder}/results_bluesky',
        f'{folder}/results_twitter',
        f'{folder}/results_reddit'
    ]
    
    # Create output directory as a subfolder of updated_folder
    OUTPUT_DIR = f"{folder}/SOTA_plots_consolidated"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Unified Figure Generation Script (CONSOLIDATED FORMAT)")
    print("=" * 60)
    print(f"Folder: {folder}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Datasets: {DATASETS}")
    print("Configuration filter: baseline + persona only (no ft, no style, no context)")
    print("Data source: postprocessing_consolidated/ and results_consolidated/")
    
    # Generate all 4 figures (pass DATASETS and OUTPUT_DIR if needed)
    generate_best_model_performance(DATASETS, OUTPUT_DIR)
    generate_first_response_similarity(DATASETS, OUTPUT_DIR)
    generate_ml_explainability_heatmap(DATASETS, OUTPUT_DIR)
    generate_aggregated_feature_frequency(DATASETS, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output files in: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SOTA figures from results')
    parser.add_argument('--folder', type=str, default='results',
                        help='Base folder path for results (default: results)')
    
    args = parser.parse_args()
    main(folder=args.folder)