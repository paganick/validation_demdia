#!/usr/bin/env python3
"""
Configuration optimization analysis script.
Generates 7 key figures:
1. SOTA vs Best Configuration performance
2. Step-wise intervention improvements
3. Response overlap summary across selection methods
4. Configuration consistency with baseline across response types
5. Cosine similarity boxplot comparison (SOTA vs Best)
6. Cosine similarity boxplot comparison (all methods)
7. Empath feature frequency by model (aggregated across datasets)

Also saves best configuration details to CSV.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle, Patch
import warnings
warnings.filterwarnings('ignore')

# Import custom utilities
from plotting_utils import (
    MODEL_PALETTE, DATASET_PALETTE, parse_filename, make_label,
    get_ordered_models, format_dataset_name, with_plot_style,
    parse_confusion_matrix, calculate_metrics, find_confusion_matrix_fields,
    get_confusion_matrix_value, normalize_dataset_name, get_dataset_color
)

# === Configuration ===
DATASETS = ['results_validation/results_bluesky', 'results_validation/results_twitter', 'results_validation/results_reddit']
OUTPUT_DIR = "configuration_optimization_figures_validation"
PRESENTATION_MODE = False
SAVE_FORMAT = 'png'

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Normalize dataset names
NORMALIZED_DATASETS = [normalize_dataset_name(d) for d in DATASETS]

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
                
                # Extract validation data type
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

                                # Track if we've found each response type to avoid duplicates
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

                                    # Only take the FIRST match for each response type
                                    if not found_random and resp_text == random_response:
                                        rows_random.append(base_data.copy())
                                        found_random = True

                                    if not found_ml and resp_text == ml_response:
                                        rows_ml.append(base_data.copy())
                                        found_ml = True

                                    if not found_cosine and resp_text == cosine_response:
                                        rows_cosine.append(base_data.copy())
                                        found_cosine = True

                                    # Early exit if all found
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
def save_best_configurations(df, response_type="random", metric="accuracy"):
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
    
    output_path = os.path.join(OUTPUT_DIR, f"best_configurations_{response_type}_{metric}.csv")
    df_best_configs.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return df_best_configs

# === Figure 1: SOTA vs Best Configuration ===
def plot_sota_vs_best_performance(df, response_type="random", metric="accuracy"):
    """Bar plot comparing SOTA configuration with best-performing configuration."""
    print(f"\n=== Generating Figure 1: SOTA vs Best Performance ({response_type}) ===")
    
    subset = df[df["response_type"] == response_type].copy()
    
    if subset.empty:
        print(f"No data available for {response_type}")
        return
    
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    
    sota_performances = []
    best_performances = []
    
    config_markers = {
        'BL': 'o',
        'BL + PE': 's',
        'BL + PE + SE': '^',
        'BL + PE + SE + CR': 'D',
        'BL + PE + SE + CR + FT': '*'
    }
    
    for (model, dataset), group in subset.groupby(['model', 'dataset']):
        sota_group = group[(group['persona'] == 1) & 
                           (group['style'] == 0) & 
                           (group['context'] == 0) & 
                           (group['ft'] == 0)
                           ]
        
        if not sota_group.empty:
            sota_row = sota_group.iloc[0]
            sota_performances.append({
                'model': model,
                'dataset': dataset,
                'sota_accuracy': sota_row[mean_col],
                'sota_std': sota_row[std_col] if pd.notna(sota_row[std_col]) else 0
            })
        
        best_idx = group[mean_col].idxmin()
        best_row = group.loc[best_idx]
        best_performances.append({
            'model': model,
            'dataset': dataset,
            'best_accuracy': best_row[mean_col],
            'best_std': best_row[std_col] if pd.notna(best_row[std_col]) else 0,
            'best_config': best_row['short_label']
        })
    
    df_sota = pd.DataFrame(sota_performances) if sota_performances else pd.DataFrame(columns=['model', 'dataset', 'sota_accuracy', 'sota_std'])
    df_best = pd.DataFrame(best_performances)
    
    if not df_sota.empty:
        df_combined = df_best.merge(df_sota, on=['model', 'dataset'], how='left')
    else:
        df_combined = df_best.copy()
        df_combined['sota_accuracy'] = np.nan
        df_combined['sota_std'] = 0
    
    model_order = get_ordered_models(df_combined['model'].unique())
    datasets = NORMALIZED_DATASETS
    
    with with_plot_style(PRESENTATION_MODE):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        n_models = len(model_order)
        bar_width = 0.13
        x_positions = np.arange(n_models)
        
        bar_offset = 0
        for i, dataset in enumerate(datasets):
            sota_data = []
            sota_stds = []
            best_data = []
            best_stds = []
            best_configs = []
            
            for model in model_order:
                model_data = df_combined[(df_combined['model'] == model) & 
                                        (df_combined['dataset'] == dataset)]
                
                if not model_data.empty:
                    row = model_data.iloc[0]
                    
                    if pd.notna(row.get('sota_accuracy')):
                        sota_data.append(row['sota_accuracy'])
                        sota_stds.append(row['sota_std'])
                    else:
                        sota_data.append(0)
                        sota_stds.append(0)
                    
                    best_data.append(row['best_accuracy'])
                    best_stds.append(row['best_std'])
                    best_configs.append(row['best_config'])
                else:
                    sota_data.append(0)
                    sota_stds.append(0)
                    best_data.append(0)
                    best_stds.append(0)
                    best_configs.append('')
            
            dataset_color = get_dataset_color(dataset)
            dataset_label = format_dataset_name(dataset)
            
            ax.bar(x_positions + bar_offset * bar_width, sota_data, bar_width,
                   label=f'{dataset_label} (BL+PE)', color=dataset_color, alpha=1.0,
                   edgecolor='white', linewidth=0.5, yerr=sota_stds, capsize=3)
            
            best_bars = ax.bar(x_positions + (bar_offset + 1) * bar_width, best_data, bar_width,
                              label=f'{dataset_label} (Best)', color=dataset_color, alpha=0.5,
                              edgecolor='white', linewidth=0.5, yerr=best_stds, capsize=3)
            
            for j, (bar, config, acc, std) in enumerate(zip(best_bars, best_configs, best_data, best_stds)):
                if config and acc > 0:
                    marker = config_markers.get(config, 'x')
                    x_pos = bar.get_x() + bar.get_width() / 2
                    y_pos = acc + std + 0.015
                    ax.plot(x_pos, y_pos, marker=marker, color='black',
                           markersize=8, markeredgewidth=1.5, markerfacecolor='white')
            
            bar_offset += 2
        
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=16)
        ax.tick_params(axis='y', labelsize=15)
        
        total_bar_width = len(datasets) * 2 * bar_width
        ax.set_xticks(x_positions + (total_bar_width - bar_width) / 2)
        ax.set_xticklabels(model_order, rotation=45, ha='right')
        
        for tick, model in zip(ax.get_xticklabels(), model_order):
            tick.set_color(MODEL_PALETTE.get(model, '#000000'))
            tick.set_fontsize(16)
        
        legend1 = ax.legend(loc='lower left', fontsize=15, ncol=3)
        ax.add_artist(legend1)
        
        marker_handles = [
            plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='white',
                      markeredgecolor='black', markersize=8, label=config, linewidth=0)
            for config, marker in config_markers.items()
        ]
        ax.legend(handles=marker_handles, loc='lower right', fontsize=15, 
                 title='Best Configuration', ncol=1, title_fontsize=15)
        
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'config_sota_vs_best_{response_type}.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")

# === Figure 2: Step-wise Intervention ===
def plot_intervention_steps_aggregated(df, response_type="random", metric="accuracy"):
    """Step-wise intervention analysis."""
    print(f"\n=== Generating Figure 2: Step-wise Intervention ({response_type}) ===")
    
    subset = df[df["response_type"] == response_type].copy()
    
    if subset.empty:
        print(f"No data available for {response_type}")
        return
    
    agg_data = subset.groupby(['model', 'style', 'context', 'ft', 'persona',
                               'response_type', 'short_label']).agg({
        'accuracy_mean': 'mean',
        'accuracy_std': 'mean'
    }).reset_index()
    
    step_comparisons = [
        ("BL", "BL + PE", "Add Persona"),
        ("BL + PE", "BL + PE + SE", "Add Style"),
        ("BL + PE + SE", "BL + PE + SE + CR", "Add Context"),
        ("BL + PE + SE + CR", "BL + PE + SE + CR + FT", "Add Fine-tuning")
    ]
    
    mean_col = f"{metric}_mean"
    model_order = get_ordered_models(agg_data["model"].unique())
    
    step_improvements = []
    
    for from_config, to_config, step_name in step_comparisons:
        improvements = []
        models_with_data = []
        
        for model in model_order:
            from_data = agg_data[(agg_data['model'] == model) & 
                                (agg_data['short_label'] == from_config)]
            to_data = agg_data[(agg_data['model'] == model) & 
                              (agg_data['short_label'] == to_config)]
            
            if len(from_data) == 1 and len(to_data) == 1:
                improvement = to_data[mean_col].iloc[0] - from_data[mean_col].iloc[0]
                improvements.append(improvement)
                models_with_data.append(model)
        
        if improvements:
            step_improvements.append({
                'step_name': step_name,
                'improvements': improvements,
                'models': models_with_data
            })
    
    if not step_improvements:
        print("  No data available")
        return
    
    with with_plot_style(PRESENTATION_MODE):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x_positions = np.arange(len(step_improvements))
        bar_width = 0.8 / len(model_order) if model_order else 0.1
        
        for model_idx, model in enumerate(model_order):
            model_improvements = []
            x_pos = []
            
            for step_idx, step_data in enumerate(step_improvements):
                if model in step_data['models']:
                    model_pos = step_data['models'].index(model)
                    model_improvements.append(step_data['improvements'][model_pos])
                else:
                    model_improvements.append(0)
                x_pos.append(step_idx + (model_idx - len(model_order)/2 + 0.5) * bar_width)
            
            if any(imp != 0 for imp in model_improvements):
                ax.bar(x_pos, model_improvements, bar_width,
                      label=model, color=MODEL_PALETTE.get(model, '#666666'), 
                      alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylabel(f'Accuracy Change', fontsize=16)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([s['step_name'] for s in step_improvements], 
                          rotation=45, ha='right', fontsize=16)
        ax.tick_params(axis='y', labelsize=15)
        ax.legend(loc='lower left', fontsize=15)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'config_stepwise_{response_type}.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")

# === Figure 3: Response Overlap Summary ===
def analyze_response_overlap(folder_paths):
    """Analyze overlap between response types."""
    print(f"\n=== Analyzing Response Overlap ===")
    
    overlap_results = []
    
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            continue
        
        dataset_name = normalize_dataset_name(folder_path)
        root_path = Path(folder_path)
        csv_files = list(root_path.rglob("*optimal_response.csv"))
        
        for filepath in csv_files:
            try:
                model, ft, context, style, persona = parse_filename(str(filepath))
                
                if model is None:
                    continue
                
                df = pd.read_csv(filepath)
                
                required_cols = ['response', 'ML_best_response', 'cosine_best_response']
                if not all(col in df.columns for col in required_cols):
                    continue
                
                total = len(df)
                if total == 0:
                    continue
                
                random_ml = (df['response'].fillna('').str.strip().str.lower() ==
                            df['ML_best_response'].fillna('').str.strip().str.lower()).sum()
                random_cosine = (df['response'].fillna('').str.strip().str.lower() ==
                                df['cosine_best_response'].fillna('').str.strip().str.lower()).sum()
                ml_cosine = (df['ML_best_response'].fillna('').str.strip().str.lower() ==
                            df['cosine_best_response'].fillna('').str.strip().str.lower()).sum()
                
                overlap_results.append({
                    'model': model,
                    'dataset': dataset_name,
                    'random_ml_pct': (random_ml / total) * 100,
                    'random_cosine_pct': (random_cosine / total) * 100,
                    'ml_cosine_pct': (ml_cosine / total) * 100
                })
                
            except Exception as e:
                continue
    
    return pd.DataFrame(overlap_results) if overlap_results else None

def plot_response_overlap_summary(df_overlap):
    """Create overlap summary plot."""
    print(f"\n=== Generating Figure 3: Response Overlap Summary ===")
    
    if df_overlap is None or df_overlap.empty:
        print("  No overlap data")
        return
    
    summary_data = []
    for (model, dataset), group in df_overlap.groupby(['model', 'dataset']):
        summary_data.append({
            'model': model,
            'dataset': dataset,
            'avg_random_ml': group['random_ml_pct'].mean(),
            'avg_random_cosine': group['random_cosine_pct'].mean(),
            'avg_ml_cosine': group['ml_cosine_pct'].mean()
        })
    
    df_summary = pd.DataFrame(summary_data)
    model_order = get_ordered_models(df_summary['model'].unique())
    datasets = NORMALIZED_DATASETS
    
    with with_plot_style(PRESENTATION_MODE):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        n_models = len(model_order)
        bar_width = 0.07
        x_positions = np.arange(n_models)
        bar_offset = 0
        
        for i, dataset in enumerate(datasets):
            dataset_color = get_dataset_color(dataset)
            dataset_label = format_dataset_name(dataset)
            
            comparisons = [
                ('avg_random_ml', 'First response - ML optimal', 0.9),
                ('avg_random_cosine', 'First Response - Cosine Optimal', 0.7),
                ('avg_ml_cosine', 'ML Optimal - Cosine Optimal', 0.5)
            ]
            
            for metric, comp_label, alpha in comparisons:
                data = []
                for model in model_order:
                    model_data = df_summary[
                        (df_summary['model'] == model) &
                        (df_summary['dataset'] == dataset)
                    ]
                    data.append(model_data[metric].iloc[0] if not model_data.empty else 0)
                
                ax.bar(x_positions + bar_offset * bar_width, data, bar_width,
                      label=f'{dataset_label} ({comp_label})', 
                      color=dataset_color, alpha=alpha,
                      edgecolor='white', linewidth=0.5)
                bar_offset += 1
        
        ax.set_ylabel('Average Overlap %', fontsize=16)
        ax.tick_params(axis='y', labelsize=15)
        
        total_bar_width = len(datasets) * 3 * bar_width
        ax.set_xticks(x_positions + (total_bar_width - bar_width) / 2)
        ax.set_xticklabels(model_order, rotation=45, ha='right')
        
        for tick, model in zip(ax.get_xticklabels(), model_order):
            tick.set_color(MODEL_PALETTE.get(model, '#000000'))
            tick.set_fontsize(15)
        
        ax.legend(loc='upper left', fontsize=11, ncol=3)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'config_overlap_summary.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")

# === Figure 4: Configuration Consistency ===
def plot_configuration_consistency_with_baseline(df, metric="accuracy"):
    """Compare baseline and best config across response types."""
    print(f"\n=== Generating Figure 4: Configuration Consistency ===")
    
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    
    random_subset = df[df["response_type"] == "random"].copy()
    best_configs_map = {}
    
    for (model, dataset), group in random_subset.groupby(['model', 'dataset']):
        best_idx = group[mean_col].idxmin()
        best_row = group.loc[best_idx]
        best_configs_map[(model, dataset)] = {
            'persona': best_row['persona'],
            'style': best_row['style'],
            'context': best_row['context'],
            'ft': best_row['ft'],
                        'short_label': best_row['short_label']
        }
    
    response_types = ['random', 'cosine_optimal', 'ML_optimal']
    comparison_data = []
    
    for (model, dataset), best_config in best_configs_map.items():
        config_data = {
            'model': model, 
            'dataset': dataset, 
            'best_config': best_config['short_label']
        }
        
        for response_type in response_types:
            rt_subset = df[df["response_type"] == response_type].copy()
            
            baseline = rt_subset[
                (rt_subset['model'] == model) &
                (rt_subset['dataset'] == dataset) &
                (rt_subset['persona'] == 1) &
                (rt_subset['style'] == 0) &
                (rt_subset['context'] == 0) &
                (rt_subset['ft'] == 0)
            ]
            
            if not baseline.empty:
                row = baseline.iloc[0]
                config_data[f'{response_type}_baseline_accuracy'] = row[mean_col]
                config_data[f'{response_type}_baseline_std'] = row[std_col] if pd.notna(row[std_col]) else 0
            else:
                config_data[f'{response_type}_baseline_accuracy'] = None
                config_data[f'{response_type}_baseline_std'] = 0
            
            matching = rt_subset[
                (rt_subset['model'] == model) &
                (rt_subset['dataset'] == dataset) &
                (rt_subset['persona'] == best_config['persona']) &
                (rt_subset['style'] == best_config['style']) &
                (rt_subset['context'] == best_config['context']) &
                (rt_subset['ft'] == best_config['ft'])
            ]
            
            if not matching.empty:
                row = matching.iloc[0]
                config_data[f'{response_type}_best_accuracy'] = row[mean_col]
                config_data[f'{response_type}_best_std'] = row[std_col] if pd.notna(row[std_col]) else 0
            else:
                config_data[f'{response_type}_best_accuracy'] = None
                config_data[f'{response_type}_best_std'] = 0
        
        comparison_data.append(config_data)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    if df_comparison.empty:
        print("  No comparison data")
        return
    
    model_order = get_ordered_models(df_comparison['model'].unique())
    datasets = NORMALIZED_DATASETS
    
    with with_plot_style(PRESENTATION_MODE):
        fig, ax = plt.subplots(figsize=(16, 8))
        
        n_models = len(model_order)
        bar_width = 0.06
        x_positions = np.arange(n_models)
        bar_offset = 0
        
        for i, dataset in enumerate(datasets):
            dataset_color = get_dataset_color(dataset)
            dataset_label = format_dataset_name(dataset)
            
            baseline_data = []
            baseline_stds = []
            
            for model in model_order:
                model_data = df_comparison[
                    (df_comparison['model'] == model) & 
                    (df_comparison['dataset'] == dataset)
                ]
                
                if not model_data.empty:
                    row = model_data.iloc[0]
                    acc = row.get('random_baseline_accuracy')
                    std = row.get('random_baseline_std', 0)
                    
                    baseline_data.append(acc if pd.notna(acc) else 0)
                    baseline_stds.append(std if pd.notna(std) else 0)
                else:
                    baseline_data.append(0)
                    baseline_stds.append(0)
            
            ax.bar(
                x_positions + bar_offset * bar_width, 
                baseline_data, 
                bar_width,
                label=f'{dataset_label} (BL+PE)',
                color=dataset_color,
                alpha=1.0,
                edgecolor='white',
                linewidth=0.5,
                yerr=baseline_stds,
                capsize=2
            )
            bar_offset += 1
            
            for j, response_type in enumerate(response_types):
                acc_data = []
                std_data = []
                
                for model in model_order:
                    model_data = df_comparison[
                        (df_comparison['model'] == model) & 
                        (df_comparison['dataset'] == dataset)
                    ]
                    
                    if not model_data.empty:
                        row = model_data.iloc[0]
                        acc = row.get(f'{response_type}_best_accuracy')
                        std = row.get(f'{response_type}_best_std', 0)
                        
                        acc_data.append(acc if pd.notna(acc) else 0)
                        std_data.append(std if pd.notna(std) else 0)
                    else:
                        acc_data.append(0)
                        std_data.append(0)
                
                alpha_map = {'random': 0.7, 'cosine_optimal': 0.5, 'ML_optimal': 0.3}
                alpha = alpha_map.get(response_type, 0.8)
                
                if response_type == 'ML_optimal':
                    response_label = 'Best + ML Optimal'
                elif response_type == 'cosine_optimal':
                    response_label = 'Best + Cosine Optimal'
                else:
                    response_label = 'Best config'
                
                ax.bar(
                    x_positions + bar_offset * bar_width, 
                    acc_data, 
                    bar_width,
                    label=f'{dataset_label} ({response_label})',
                    color=dataset_color,
                    alpha=alpha,
                    edgecolor='white',
                    linewidth=0.5,
                    yerr=std_data,
                    capsize=3
                )
                bar_offset += 1
        
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=16)
        ax.tick_params(axis='y', labelsize=15)
        
        total_bar_width = len(datasets) * 4 * bar_width
        ax.set_xticks(x_positions + (total_bar_width - bar_width) / 2)
        ax.set_xticklabels(model_order, rotation=45, ha='right')
        
        for tick, model in zip(ax.get_xticklabels(), model_order):
            tick.set_color(MODEL_PALETTE.get(model, '#000000'))
            tick.set_fontsize(15)
        
        ax.legend(loc='lower right', fontsize=12, ncol=2, 
                 title='Dataset & Configuration', title_fontsize=12)
        
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'config_consistency_with_baseline.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")

# === Figure 5: Cosine Similarity Boxplot (SOTA vs Best) ===
def plot_cosine_similarity_boxplot(similarity_data, df_best_configs):
    """Create boxplot comparing SOTA vs Best config for random response type."""
    print(f"\n=== Generating Figure 5: Cosine Similarity Boxplot (SOTA vs Best) ===")
    
    df_random = similarity_data['random']
    
    if df_random.empty:
        print("  No random similarity data")
        return
    
    # SOTA data
    df_sota = df_random[(df_random['persona'] == 1) & 
                        (df_random['style'] == 0) & 
                        (df_random['context'] == 0) & 
                        (df_random['ft'] == 0)
                        ].copy()
    df_sota['type'] = 'SOTA (BL+PE)'
    
    # Best config data
    best_data_list = []
    for _, config_row in df_best_configs.iterrows():
        df_filtered = df_random[
            (df_random['model'] == config_row['model']) &
            (df_random['dataset'] == config_row['dataset']) &
            (df_random['persona'] == int(config_row['has_persona'])) &
            (df_random['style'] == int(config_row['has_style'])) &
            (df_random['context'] == int(config_row['has_context'])) &
            (df_random['ft'] == int(config_row['has_finetuning']))
        ].copy()
        
        if len(df_filtered) > 0:
            df_filtered['type'] = 'Best Config'
            best_data_list.append(df_filtered)
    
    if not best_data_list:
        print("  No best config data")
        return
    
    df_best = pd.concat(best_data_list, ignore_index=True)
    df_combined = pd.concat([df_sota, df_best], ignore_index=True)
    
    datasets = NORMALIZED_DATASETS
    
    with with_plot_style(PRESENTATION_MODE):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        box_width = 0.35
        group_spacing = 0.4
        positions_per_dataset = []
        
        for i in range(len(datasets)):
            base = i * (2 * box_width + group_spacing)
            positions_per_dataset.append([base, base + box_width])
        
        all_positions = [pos for dataset_pos in positions_per_dataset for pos in dataset_pos]
        
        all_data = []
        all_colors = []
        all_alphas = []
        
        for dataset in datasets:
            sota_subset = df_combined[(df_combined['dataset'] == dataset) & 
                                     (df_combined['type'] == 'SOTA (BL+PE)')]
            all_data.append(sota_subset['similarity'].values if len(sota_subset) > 0 else [])
            all_colors.append(get_dataset_color(dataset))
            all_alphas.append(1.0)
            
            best_subset = df_combined[(df_combined['dataset'] == dataset) & 
                                     (df_combined['type'] == 'Best Config')]
            all_data.append(best_subset['similarity'].values if len(best_subset) > 0 else [])
            all_colors.append(get_dataset_color(dataset))
            all_alphas.append(0.5)
        
        bp = ax.boxplot(
            all_data,
            positions=all_positions,
            widths=box_width * 0.8,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color='red', linewidth=2),
            flierprops=dict(marker='o', markersize=3, alpha=0.4)
        )
        
        for patch, color, alpha in zip(bp['boxes'], all_colors, all_alphas):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)
        
        for i, (color, alpha) in enumerate(zip(all_colors, all_alphas)):
            bp['whiskers'][i*2].set_color(color)
            bp['whiskers'][i*2+1].set_color(color)
            bp['whiskers'][i*2].set_alpha(alpha)
            bp['whiskers'][i*2+1].set_alpha(alpha)
            bp['caps'][i*2].set_color(color)
            bp['caps'][i*2+1].set_color(color)
            bp['caps'][i*2].set_alpha(alpha)
            bp['caps'][i*2+1].set_alpha(alpha)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        x_tick_positions = [np.mean(positions) for positions in positions_per_dataset]
        dataset_labels = [format_dataset_name(d) for d in datasets]
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(dataset_labels, rotation=0, ha='center')
        
        for tick, dataset in zip(ax.get_xticklabels(), datasets):
            tick.set_color(get_dataset_color(dataset))
            tick.set_fontsize(16)
        
        ax.set_ylim([-1, 1])
        ax.set_ylabel("Cosine Similarity", fontsize=16)
        ax.tick_params(axis='y', labelsize=15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        legend_elements = [
            Patch(facecolor='gray', alpha=1.0, edgecolor='black', label='SOTA (BL+PE)'),
            Patch(facecolor='gray', alpha=0.5, edgecolor='black', label='Best Config')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=15, 
                 title='Configuration Type', title_fontsize=15)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'sota_vs_best_cosine_similarity_boxplot.png')
        fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=PRESENTATION_MODE)
        plt.close()
        print(f"  Saved to: {output_path}")

# === Figure 6: Cosine Similarity Boxplot (All Methods) ===
def plot_cosine_similarity_boxplot_all_methods(similarity_data, df_best_configs):
    """Create boxplot comparing baseline and best across all response types."""
    print(f"\n=== Generating Figure 6: Cosine Similarity Boxplot (All Methods) ===")
    
    df_random = similarity_data['random']
    df_ml = similarity_data['ML_optimal']
    df_cosine = similarity_data['cosine_optimal']
    
    if df_random.empty or df_ml.empty or df_cosine.empty:
        print("  Missing data for some response types")
        return
    
    # Baseline from random
    df_baseline = df_random[(df_random['persona'] == 1) & 
                            (df_random['style'] == 0) & 
                            (df_random['context'] == 0) & 
                            (df_random['ft'] == 0)
                            ].copy()
    df_baseline['type'] = 'Baseline (BL+PE)'
    
    # Best configs for each response type
    response_datasets = [
        (df_random, 'Best Random'),
        (df_cosine, 'Best Cosine Optimal'),
        (df_ml, 'Best ML Optimal')
    ]
    
    all_best_data = []
    
    for df_source, type_label in response_datasets:
        best_data_list = []
        
        for _, config_row in df_best_configs.iterrows():
            df_filtered = df_source[
                (df_source['model'] == config_row['model']) &
                (df_source['dataset'] == config_row['dataset']) &
                (df_source['persona'] == int(config_row['has_persona'])) &
                (df_source['style'] == int(config_row['has_style'])) &
                (df_source['context'] == int(config_row['has_context'])) &
                (df_source['ft'] == int(config_row['has_finetuning']))
            ].copy()
            
            if len(df_filtered) > 0:
                df_filtered['type'] = type_label
                best_data_list.append(df_filtered)
        
        if best_data_list:
            all_best_data.append(pd.concat(best_data_list, ignore_index=True))
    
    if not all_best_data:
        print("  No best config data")
        return
    
    df_combined = pd.concat([df_baseline] + all_best_data, ignore_index=True)
    datasets = NORMALIZED_DATASETS
    
    with with_plot_style(PRESENTATION_MODE):
        fig, ax = plt.subplots(figsize=(16, 8))
        
        box_width = 0.20
        group_spacing = 0.3
        positions_per_dataset = []
        
        for i in range(len(datasets)):
            base = i * (4 * box_width + group_spacing)
            positions_per_dataset.append([
                base,
                base + box_width,
                base + 2 * box_width,
                base + 3 * box_width
            ])
        
        all_positions = [pos for dataset_pos in positions_per_dataset for pos in dataset_pos]
        
        all_data = []
        all_colors = []
        all_alphas = []
        
        config_types = [
            ('Baseline (BL+PE)', 1.0),
            ('Best Random', 0.7),
            ('Best Cosine Optimal', 0.5),
            ('Best ML Optimal', 0.3)
        ]
        
        for dataset in datasets:
            for config_type, alpha in config_types:
                subset = df_combined[(df_combined['dataset'] == dataset) & 
                                    (df_combined['type'] == config_type)]
                all_data.append(subset['similarity'].values if len(subset) > 0 else [])
                all_colors.append(get_dataset_color(dataset))
                all_alphas.append(alpha)
        
        bp = ax.boxplot(
            all_data,
            positions=all_positions,
            widths=box_width * 0.8,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color='red', linewidth=2),
            flierprops=dict(marker='o', markersize=3, alpha=0.4)
        )
        
        for patch, color, alpha in zip(bp['boxes'], all_colors, all_alphas):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)
        
        for i, (color, alpha) in enumerate(zip(all_colors, all_alphas)):
            bp['whiskers'][i*2].set_color(color)
            bp['whiskers'][i*2+1].set_color(color)
            bp['whiskers'][i*2].set_alpha(alpha)
            bp['whiskers'][i*2+1].set_alpha(alpha)
            bp['caps'][i*2].set_color(color)
            bp['caps'][i*2+1].set_color(color)
            bp['caps'][i*2].set_alpha(alpha)
            bp['caps'][i*2+1].set_alpha(alpha)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        x_tick_positions = [np.mean(positions) for positions in positions_per_dataset]
        dataset_labels = [format_dataset_name(d) for d in datasets]
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(dataset_labels, rotation=0, ha='center')
        
        for tick, dataset in zip(ax.get_xticklabels(), datasets):
            tick.set_color(get_dataset_color(dataset))
            tick.set_fontsize(16)
        
        ax.set_ylim([-1, 1])
        ax.set_ylabel("Cosine Similarity", fontsize=16)
        ax.tick_params(axis='y', labelsize=15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        legend_elements = [
            Patch(facecolor='gray', alpha=1.0, edgecolor='black', label='Baseline (BL+PE)'),
            Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='Best config'),
            Patch(facecolor='gray', alpha=0.5, edgecolor='black', label='Best config + Cosine Optimal'),
            Patch(facecolor='gray', alpha=0.3, edgecolor='black', label='Best config + ML Optimal')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=15, 
                 title='Configuration Type', title_fontsize=15)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'sota_vs_best_cosine_similarity_boxplot_all_methods.png')
        fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=PRESENTATION_MODE)
        plt.close()
        print(f"  Saved to: {output_path}")

# === Figure 7: Empath Feature Frequency (NEW) ===
def load_empath_feature_data(folder_paths, df_best_configs, response_type='random'):
    """
    Load Empath feature data for optimal configurations only.
    Uses the best_configurations dataframe to filter.
    """
    print(f"\n=== Loading Empath Feature Data ===")
    
    feature_model_counts = {}  # {feature: {model: count}}
    
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            continue
        
        dataset_name = normalize_dataset_name(folder_path)
        print(f"\nProcessing {dataset_name}...")
        
        # Get optimal configs for this dataset
        dataset_configs = df_best_configs[df_best_configs['dataset'] == dataset_name]
        
        root_path = Path(folder_path)
        
        for _, config_row in dataset_configs.iterrows():
            model = config_row['model']
            
            # Build expected configuration parameters
            expected_config = {
                'ft': int(config_row['has_finetuning']),
                'context': int(config_row['has_context']),
                'style': int(config_row['has_style']),
                                'persona': int(config_row['has_persona'])
            }
            
            # Search for matching empath files
            for root, _, files in os.walk(root_path):
                for file in files:
                    if not file.endswith("empath_significant_features.csv"):
                        continue
                    
                    # Match only relevant response type
                    if response_type == "random" and "random_validation" not in file:
                        continue
                    
                    filepath = os.path.join(root, file)
                    
                    try:
                        file_model, file_ft, file_context, file_style, file_persona = parse_filename(file)
                        
                        # Check if this file matches the optimal configuration
                        if (file_model == model and
                            file_ft == expected_config['ft'] and
                            file_context == expected_config['context'] and
                            file_style == expected_config['style'] and

                            file_persona == expected_config['persona']):
                            
                            # Load and process the file
                            df = pd.read_csv(filepath)
                            sig_features = df[df["adjusted_p_value"] < 0.05]["feature"].tolist()
                            
                            # Count each significant feature by model
                            for feature in sig_features:
                                clean_feature = feature.replace("_", " ")
                                if clean_feature not in feature_model_counts:
                                    feature_model_counts[clean_feature] = {}
                                if model not in feature_model_counts[clean_feature]:
                                    feature_model_counts[clean_feature][model] = 0
                                feature_model_counts[clean_feature][model] += 1
                            
                            print(f"  Loaded: {model} - {config_row['best_config_short']}")
                            
                    except Exception as e:
                        continue
    
    if not feature_model_counts:
        print("  No Empath feature data found!")
        return None
    
    print(f"\nFound {len(feature_model_counts)} unique features")
    return feature_model_counts

def plot_empath_feature_frequency(folder_paths, df_best_configs, response_type='random', top_n=20):
    """
    Create aggregated plot with 3 subplots showing feature frequency by model 
    for optimal configurations, one subplot per dataset.
    """
    print(f"\n=== Generating Figure 7: Empath Feature Frequency ===")
    
    # Collect data per dataset (similar to original's all_data list)
    all_data = []
    
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            continue
        
        dataset_name = normalize_dataset_name(folder_path)
        print(f"\nProcessing {dataset_name}...")
        
        # Get optimal configs for this dataset
        dataset_configs = df_best_configs[df_best_configs['dataset'] == dataset_name]
        
        print(f"Dataset name: {dataset_name}")
        print(f"Configs found: {len(dataset_configs)}")

        feature_model_counts = {}  # {feature: {model: count}} for this dataset
        root_path = Path(folder_path)
        
        for _, config_row in dataset_configs.iterrows():
            model = config_row['model']
            
            # Build expected configuration parameters
            expected_config = {
                'ft': int(config_row['has_finetuning']),
                'context': int(config_row['has_context']),
                'style': int(config_row['has_style']),
                                'persona': int(config_row['has_persona'])
            }
            
            # Search for matching empath files
            for root, _, files in os.walk(root_path):
                for file in files:
                    if not file.endswith("empath_significant_features.csv"):
                        continue
                    
                    # Match only relevant response type
                    if response_type == "random" and "random_validation" not in file:
                        continue
                    
                    filepath = os.path.join(root, file)
                    
                    try:
                        file_model, file_ft, file_context, file_style, file_persona = parse_filename(file)
                        
                        # Check if this file matches the optimal configuration
                        if (file_model == model and
                            file_ft == expected_config['ft'] and
                            file_context == expected_config['context'] and
                            file_style == expected_config['style'] and

                            file_persona == expected_config['persona']):
                            
                            # Load and process the file
                            df = pd.read_csv(filepath)
                            sig_features = df[df["adjusted_p_value"] < 0.05]["feature"].tolist()
                            
                            # Count each significant feature by model
                            for feature in sig_features:
                                clean_feature = feature.replace("_", " ")
                                if clean_feature not in feature_model_counts:
                                    feature_model_counts[clean_feature] = {}
                                if model not in feature_model_counts[clean_feature]:
                                    feature_model_counts[clean_feature][model] = 0
                                feature_model_counts[clean_feature][model] += 1
                            
                            print(f"  Loaded: {model} - {config_row['best_config_short']}")
                            
                    except Exception as e:
                        continue
        
        # Calculate total counts per feature and sort
        if feature_model_counts:
            feature_totals = {}
            for feature, model_counts in feature_model_counts.items():
                feature_totals[feature] = sum(model_counts.values())
            
            # Sort by total count and take top N
            sorted_features = sorted(feature_totals.items(), key=lambda x: x[1], reverse=True)
            top_features = [feature for feature, _ in sorted_features[:top_n]]
            
            # Store data for this dataset
            all_data.append({
                'features': top_features,
                'model_counts': feature_model_counts,
                'dataset': format_dataset_name(dataset_name)
            })
    
    if not all_data:
        print("  No feature data to plot")
        return
    
    # Get all unique models across all datasets for consistent legend
    all_models = set()
    for data in all_data:
        for feature_counts in data['model_counts'].values():
            all_models.update(feature_counts.keys())
    
    ordered_models = get_ordered_models(list(all_models))
    total_models = len(ordered_models)
    
    # Create aggregated plot with 3 subplots (one per dataset)
    fig_width = 18
    fig_height = 8
    
    with with_plot_style(PRESENTATION_MODE):
        fig, axes = plt.subplots(1, len(all_data), figsize=(fig_width, fig_height), 
                                sharey=False)
        
        if len(all_data) == 1:
            axes = [axes]
        
        for i, (ax, data) in enumerate(zip(axes, all_data)):
            features = data['features']
            model_counts = data['model_counts']
            dataset_name = data['dataset']
            
            # Get models present in THIS specific dataset
            dataset_models = set()
            for feature_counts in model_counts.values():
                dataset_models.update(feature_counts.keys())
            
            # Create stacked horizontal bar plot with percentages
            bottom = np.zeros(len(features))
            
            for model in ordered_models:
                if model in dataset_models:
                    model_values = [model_counts.get(feature, {}).get(model, 0) for feature in features]
                    # Convert to percentage of total models
                    model_percentages = [(value / total_models * 100) for value in model_values]
                    color = MODEL_PALETTE.get(model, 'gray')
                    
                    bars = ax.barh(range(len(features)), model_percentages, left=bottom, 
                                color=color, alpha=0.7, label=model)
                    bottom += model_percentages
            
            # Format axes
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('Share (%)', fontsize=12)
            
            # Color the title by dataset
            dataset_color = get_dataset_color(dataset_name)
            ax.set_title(dataset_name, fontsize=14, pad=10, color=dataset_color)
            ax.set_xlim(0, 105)
            
            # Invert y-axis to show highest counts at top
            ax.invert_yaxis()
            
            # Add grid for better readability
            ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Create model legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=MODEL_PALETTE.get(model, 'gray'), 
                                       alpha=0.7, label=model) for model in ordered_models]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.1), 
                  ncol=min(len(ordered_models), 4), fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.2)
        
        # Save output
        output_path = os.path.join(OUTPUT_DIR, f'empath_feature_frequency_top{top_n}.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close()
        
        print(f"  Saved to: {output_path}")
        print(f"  Total models/configurations: {total_models}")

# === Load Feature Importance Data ===
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
        
        # Get optimal configs for this dataset
        dataset_configs = df_best_configs[df_best_configs['dataset'] == dataset_name]
        
        root_path = Path(folder_path)
        
        for _, config_row in dataset_configs.iterrows():
            model = config_row['model']
            
            # Build expected configuration parameters
            expected_config = {
                'ft': int(config_row['has_finetuning']),
                'context': int(config_row['has_context']),
                'style': int(config_row['has_style']),
                                'persona': int(config_row['has_persona'])
            }
            
            # Search for matching feature importance files
            for root, _, files in os.walk(root_path):
                for file in files:
                    if not file.endswith("feature_correlation_stats.csv"):
                        continue
                    
                    # Match only relevant response type
                    if response_type == "random" and "random_validation" not in file:
                        continue
                    
                    filepath = os.path.join(root, file)
                    
                    try:
                        file_model, file_ft, file_context, file_style, file_persona = parse_filename(file)
                        
                        # Check if this file matches the optimal configuration
                        if (file_model == model and
                            file_ft == expected_config['ft'] and
                            file_context == expected_config['context'] and
                            file_style == expected_config['style'] and

                            file_persona == expected_config['persona']):
                            
                            # Load and process the file
                            df = pd.read_csv(filepath)
                            
                            # Add metadata
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


# === Helper function for feature name formatting ===
def format_feature_name(feature):
    """Format feature names for display."""
    return feature.replace('_', ' ').title()


# === Figure 8: Feature Importance Heatmap ===
def plot_feature_importance_heatmap(df, max_features=10):
    """
    Create three horizontally aligned heatmaps showing feature importance 
    across datasets for optimal configurations.
    """
    print(f"\n=== Generating Figure 8: Feature Importance Heatmap ===")
    
    if df is None or df.empty:
        print("  No feature importance data to plot")
        return
    
    with with_plot_style(PRESENTATION_MODE):
        # Order datasets: Bluesky, Twitter, Reddit
        dataset_order = NORMALIZED_DATASETS
        datasets = [d for d in dataset_order if d in df['dataset'].unique()]
        
        # Get all models
        models = get_ordered_models(df['model'].unique())
        model_colors = [MODEL_PALETTE.get(model, 'gray') for model in models]
        
        # Calculate overall importance (for selecting top features)
        overall_importance = df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        # Get overall top features
        top_overall = set(overall_importance.head(max_features).index)
        
        # Get top 2 features per model
        top_per_model = set()
        for model in models:
            model_data = df[df['model'] == model]
            model_importance = model_data.groupby('feature')['importance'].mean().sort_values(ascending=False)
            top_per_model.update(model_importance.head(2).index)
        
        # Combine to get candidate features
        candidate_features = top_overall.union(top_per_model)
        
        print(f"  Candidate features: {len(candidate_features)}")
        
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(22, 5), sharey=True)
        
        # Calculate global min/max for consistent color scaling
        all_values = []
        for dataset in datasets:
            dataset_data = df[df['dataset'] == dataset]
            for model in models:
                model_data = dataset_data[dataset_data['model'] == model]
                if not model_data.empty:
                    model_importance = model_data.groupby('feature')['importance'].mean()
                    values = list(model_importance.values)
                    all_values.extend(values)
        
        if not all_values:
            print("  No importance values found")
            return
        
        vmin, vmax = min(all_values), max(all_values)
        print(f"  Global importance range: {vmin:.4f} to {vmax:.4f}")
        
        # Use grayscale colormap
        cmap = 'gray_r'  # white (low) to black (high)
        
        # Use PowerNorm for better color discrimination
        from matplotlib.colors import PowerNorm
        
        # Shift values to be positive if needed
        if vmin <= 0:
            shift = abs(vmin) + 0.0001
            vmin_shifted = vmin + shift
            vmax_shifted = vmax + shift
            use_shift = True
        else:
            vmin_shifted = vmin
            vmax_shifted = vmax
            use_shift = False
        
        norm = PowerNorm(gamma=1.0, vmin=vmin_shifted, vmax=vmax_shifted)
        
        # Create heatmap for each dataset
        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            
            # Filter data for this dataset
            dataset_data = df[df['dataset'] == dataset]
            
            # Calculate importance for THIS dataset only
            dataset_importance = dataset_data.groupby('feature')['importance'].mean().sort_values(ascending=False)
            
            # Get top features for this dataset from candidates
            dataset_top = set(dataset_importance.head(max_features).index)
            
            # Also ensure we include top 2 per model for this dataset
            dataset_top_per_model = set()
            for model in models:
                model_data = dataset_data[dataset_data['model'] == model]
                if not model_data.empty:
                    model_importance = model_data.groupby('feature')['importance'].mean().sort_values(ascending=False)
                    dataset_top_per_model.update(model_importance.head(2).index)
            
            # Combine and limit to candidate features
            selected_features_for_dataset = dataset_top.union(dataset_top_per_model).intersection(candidate_features)
            
            # Sort by importance within THIS dataset
            selected_features = sorted(
                selected_features_for_dataset,
                key=lambda x: dataset_importance.get(x, 0),
                reverse=True
            )[:max_features]
            
            print(f"  {dataset} - Selected {len(selected_features)} features")
            
            # Build importance matrix for this dataset
            importance_matrix = []
            for model in models:
                model_data = dataset_data[dataset_data['model'] == model]
                if not model_data.empty:
                    model_importance = model_data.groupby('feature')['importance'].mean()
                    importance_row = [model_importance.get(feature, 0) for feature in selected_features]
                else:
                    importance_row = [0] * len(selected_features)
                importance_matrix.append(importance_row)
            
            importance_array = np.array(importance_matrix)
            
            # Apply shift if needed
            if use_shift:
                importance_array_shifted = importance_array + shift
            else:
                importance_array_shifted = importance_array
            
            # Create heatmap
            im = ax.imshow(importance_array_shifted, cmap=cmap, aspect='auto', norm=norm)
            
            # Format feature names
            formatted_features = [format_feature_name(f) for f in selected_features]
            
            # Set x-axis
            ax.set_xticks(range(len(formatted_features)))
            ax.set_xticklabels(formatted_features, rotation=45, ha='right', fontsize=14)
            
            # Set y-axis (only for first subplot)
            if idx == 0:
                ax.set_yticks(range(len(models)))
                ax.set_yticklabels([])
                
                # Add colored model labels
                for i, (model, color) in enumerate(zip(models, model_colors)):
                    ax.text(-0.5, i, model, ha='right', va='center', color=color, 
                           fontsize=14)
            
            # Add values to cells
            font_size = 9
            for i in range(len(models)):
                for j in range(len(formatted_features)):
                    value = importance_array[i, j]
                    value_shifted = importance_array_shifted[i, j]
                    normalized_value = (value_shifted - vmin_shifted) / (vmax_shifted - vmin_shifted)
                    text_color = 'white' if normalized_value > 0.5 else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                           color=text_color, fontsize=font_size)
            
            # Set title with colored dataset name
            dataset_label = format_dataset_name(dataset)
            dataset_color = get_dataset_color(dataset)
            ax.set_title(dataset_label, fontsize=15, pad=10, color=dataset_color)
            
            # Remove y-axis for non-first subplots
            if idx > 0:
                ax.set_yticks([])
        
        # Add colorbar
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Feature Importance', fontsize=14)
        
        # Save output
        output_path = os.path.join(OUTPUT_DIR, f'feature_importance_heatmap_top{max_features}.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close()
        
        print(f"  Saved to: {output_path}")

# === Main Execution ===
def main():
    """Main function to generate configuration optimization figures."""
    print("=" * 60)
    print("Configuration Optimization Analysis")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Datasets: {DATASETS}")
    
    # Load accuracy data
    print("\n=== Loading Accuracy Data ===")
    df = load_configuration_data(DATASETS)
    
    if df.empty:
        print("No accuracy data found!")
        return
    
    print(f"Loaded {len(df)} records")
    
    # Save best configurations
    df_best_configs = save_best_configurations(df, response_type="random", metric="accuracy")
    
    if df_best_configs is None:
        print("Could not determine best configurations!")
        return
    
    # Generate accuracy-based figures
    plot_sota_vs_best_performance(df, response_type="random", metric="accuracy")
    plot_intervention_steps_aggregated(df, response_type="random", metric="accuracy")
    
    # Analyze and plot response overlap
    df_overlap = analyze_response_overlap(DATASETS)
    if df_overlap is not None:
        plot_response_overlap_summary(df_overlap)
    
    # Plot configuration consistency
    plot_configuration_consistency_with_baseline(df, metric="accuracy")
    
    # Load cosine similarity data
    similarity_data = load_cosine_similarities_by_response_type(DATASETS)
    
    # Generate cosine similarity figures
    if similarity_data['random'] is not None and not similarity_data['random'].empty:
        plot_cosine_similarity_boxplot(similarity_data, df_best_configs)
        plot_cosine_similarity_boxplot_all_methods(similarity_data, df_best_configs)
    
    # Generate Empath feature frequency plot
    feature_model_counts = load_empath_feature_data(DATASETS, df_best_configs, response_type='random')
    if feature_model_counts:
        plot_empath_feature_frequency(DATASETS, df_best_configs, top_n=20)
    
    # Generate feature importance heatmap
    df_importance = load_feature_importance_data(DATASETS, df_best_configs, response_type='random')
    if df_importance is not None:
        plot_feature_importance_heatmap(df_importance, max_features=10)
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output files in: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
