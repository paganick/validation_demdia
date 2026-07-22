#!/usr/bin/env python3
"""
Configuration-optimization ("best config" per model/dataset) figures that
appear in the main text:
  Figure 5  - SOTA vs Best Configuration performance
  Figure 6  - Step-wise intervention improvements
  Figure 7  - Cosine similarity boxplot (SOTA vs Best)
  Figure 8  - Feature importance heatmap (top-10, best config)
  Figure 9  - feeds plot_empath_feature_heatmap.py (Empath feature frequency)
  Figure 10 - Configuration consistency with baseline across response types
  Figure 11 - Cosine similarity boxplot (all selection methods)

Also saves best configuration details to CSV (best_configurations_*.csv),
a prerequisite consumed by several other analysis/ scripts.

For the SI-only configuration-optimization figures (response overlap, cosine
similarity by model x method, feature bias), see generate_config_SI_figures.py.

Usage:
    python analysis/generate_config_main_figures.py results_cleaned/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.src.plotting_utils import (
    MODEL_PALETTE, get_ordered_models, format_dataset_name, with_plot_style,
    normalize_dataset_name, get_dataset_color, parse_filename,
)
from analysis._config_common import (
    _save_stats, format_feature_name,
    load_configuration_data, load_cosine_similarities_by_response_type,
    save_best_configurations, load_feature_importance_data,
)

# === Configuration ===
DATASETS = ['results_cleaned_20260309_095640/bluesky']
OUTPUT_DIR = "results_revision/configuration_optimization_figures"
PRESENTATION_MODE = False
SAVE_FORMAT = 'png'
NORMALIZED_DATASETS = []  # populated in main()


# === Figure 5: SOTA vs Best Configuration ===
def plot_sota_vs_best_performance(df, response_type="random", metric="accuracy"):
    """Bar plot comparing SOTA configuration with best-performing configuration."""
    print(f"\n=== Generating Figure 5: SOTA vs Best Performance ({response_type}) ===")

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

        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=20)
        ax.tick_params(axis='y', labelsize=18)

        total_bar_width = len(datasets) * 2 * bar_width
        ax.set_xticks(x_positions + (total_bar_width - bar_width) / 2)
        ax.set_xticklabels(model_order, rotation=45, ha='right')

        for tick, model in zip(ax.get_xticklabels(), model_order):
            tick.set_color('black')
            tick.set_fontsize(20)

        legend1 = ax.legend(loc='lower left', fontsize=18, ncol=1)
        ax.add_artist(legend1)

        marker_handles = [
            plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='white',
                      markeredgecolor='black', markersize=8, label=config, linewidth=0)
            for config, marker in config_markers.items()
        ]
        ax.legend(handles=marker_handles, loc='lower right', fontsize=18,
                 title='Best Configuration', ncol=1, title_fontsize=18)

        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        stats_path = os.path.join(OUTPUT_DIR, f'config_sota_vs_best_{response_type}_stats.csv')
        _save_stats(subset, ['model', 'dataset', 'short_label'], f'{metric}_mean', stats_path)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'config_sota_vs_best_{response_type}.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")


# === Figure 6: Step-wise Intervention ===
def plot_intervention_steps_aggregated(df, response_type="random", metric="accuracy"):
    """Step-wise intervention analysis."""
    print(f"\n=== Generating Figure 6: Step-wise Intervention ({response_type}) ===")

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
                      alpha=1.0, edgecolor='black', linewidth=0.5)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylabel(f'Accuracy Change', fontsize=20)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([s['step_name'] for s in step_improvements],
                          rotation=45, ha='right', fontsize=20)
        ax.tick_params(axis='y', labelsize=18)
        legend = ax.legend(loc='upper left', fontsize=18, ncol=2)
        for text in legend.get_texts():
            text.set_color('black')

        if step_improvements:
            step_rows = []
            for s in step_improvements:
                for mdl, imp in zip(s['models'], s['improvements']):
                    step_rows.append({'step': s['step_name'], 'model': mdl, 'accuracy_change': imp})
            stats_path = os.path.join(OUTPUT_DIR, f'config_stepwise_{response_type}_stats.csv')
            pd.DataFrame(step_rows).to_csv(stats_path, index=False)
            print(f"  Stats saved to: {stats_path}")
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'config_stepwise_{response_type}.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")


# === Figure 10: Configuration Consistency ===
def plot_configuration_consistency_with_baseline(df, metric="accuracy"):
    """Compare baseline and best config across response types."""
    print(f"\n=== Generating Figure 10: Configuration Consistency ===")

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
            tick.set_color('black')
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


# === Figure 7: Cosine Similarity Boxplot (SOTA vs Best) ===
def plot_cosine_similarity_boxplot(similarity_data, df_best_configs):
    """Create boxplot comparing SOTA vs Best config for random response type."""
    print(f"\n=== Generating Figure 7: Cosine Similarity Boxplot (SOTA vs Best) ===")

    df_random = similarity_data['random']

    if df_random.empty:
        print("  No random similarity data")
        return

    df_sota = df_random[(df_random['persona'] == 1) &
                        (df_random['style'] == 0) &
                        (df_random['context'] == 0) &
                        (df_random['ft'] == 0)
                        ].copy()
    df_sota['type'] = 'Reference config (BL+PE)'

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

    stats_path = os.path.join(OUTPUT_DIR, 'config_cosine_sota_vs_best_stats.csv')
    _save_stats(df_combined, ['dataset', 'model', 'config_label', 'type'], 'similarity', stats_path)

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
                                     (df_combined['type'] == 'Reference config (BL+PE)')]
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
            tick.set_fontsize(20)

        ax.set_ylim([-1, 1])
        ax.set_ylabel("Cosine Similarity", fontsize=20)
        ax.tick_params(axis='y', labelsize=18)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        legend_elements = [
            Patch(facecolor='gray', alpha=1.0, edgecolor='black', label='Reference config (BL+PE)'),
            Patch(facecolor='gray', alpha=0.5, edgecolor='black', label='Best Config')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=18,
                 title='Configuration Type', title_fontsize=18)

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'sota_vs_best_cosine_similarity_boxplot.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=PRESENTATION_MODE)
        plt.close()
        print(f"  Saved to: {output_path}")


# === Figure 11: Cosine Similarity Boxplot (All Methods) ===
def plot_cosine_similarity_boxplot_all_methods(similarity_data, df_best_configs):
    """Create boxplot comparing baseline and best across all response types."""
    print(f"\n=== Generating Figure 11: Cosine Similarity Boxplot (All Methods) ===")

    df_random = similarity_data['random']
    df_ml = similarity_data['ML_optimal']
    df_cosine = similarity_data['cosine_optimal']

    if df_random.empty or df_ml.empty or df_cosine.empty:
        print("  Missing data for some response types")
        return

    df_baseline = df_random[(df_random['persona'] == 1) &
                            (df_random['style'] == 0) &
                            (df_random['context'] == 0) &
                            (df_random['ft'] == 0)
                            ].copy()
    df_baseline['type'] = 'Baseline (BL+PE)'

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

    stats_path = os.path.join(OUTPUT_DIR, 'config_cosine_all_methods_stats.csv')
    _save_stats(df_combined, ['dataset', 'model', 'config_label', 'type'], 'similarity', stats_path)
    summary_path = os.path.join(OUTPUT_DIR, 'config_cosine_all_methods_summary.csv')
    _save_stats(df_combined, ['dataset', 'type'], 'similarity', summary_path)

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
            tick.set_fontsize(20)

        ax.set_ylim([-1, 1])
        ax.set_ylabel("Cosine Similarity", fontsize=20)
        ax.tick_params(axis='y', labelsize=18)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        legend_elements = [
            Patch(facecolor='gray', alpha=1.0, edgecolor='black', label='Baseline (BL+PE)'),
            Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='Best config'),
            Patch(facecolor='gray', alpha=0.5, edgecolor='black', label='Best config + Cosine Optimal'),
            Patch(facecolor='gray', alpha=0.3, edgecolor='black', label='Best config + ML Optimal')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=18,
                 title='Configuration Type', title_fontsize=18)

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'sota_vs_best_cosine_similarity_boxplot_all_methods.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=PRESENTATION_MODE)
        plt.close()
        print(f"  Saved to: {output_path}")


# === Figure 9 (data prep): Empath Feature Frequency ===
def prepare_config_empath_feature_stats(folder_paths, df_best_configs, response_type='random'):
    """
    For each dataset's best configuration per model, count how many times each
    Empath feature reaches significance (adjusted p < 0.05).

    Writes config_empath_stats_{response_type}.csv (every significant feature,
    unfiltered), consumed by plot_empath_feature_heatmap.py to produce Figure 9,
    which applies its own top-N selection when reading this CSV. This function
    only prepares data -- the actual figure is the features x models heatmap
    (see plot_empath_feature_heatmap.py), which replaced this analysis's
    original stacked-bar rendering.
    """
    print(f"\n=== Preparing Figure 9 data: Empath Feature Frequency ===")

    empath_rows = []

    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            continue

        dataset_name = normalize_dataset_name(folder_path)
        print(f"\nProcessing {dataset_name}...")

        dataset_configs = df_best_configs[df_best_configs['dataset'] == dataset_name]
        print(f"Configs found: {len(dataset_configs)}")

        feature_model_counts = {}
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
                    if not file.endswith("empath_significant_features.csv"):
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
                            sig_features = df[df["adjusted_p_value"] < 0.05]["feature"].tolist()

                            for feature in sig_features:
                                clean_feature = feature.replace("_", " ")
                                feature_model_counts.setdefault(clean_feature, {})
                                feature_model_counts[clean_feature][model] = feature_model_counts[clean_feature].get(model, 0) + 1

                            print(f"  Loaded: {model} - {config_row['best_config_short']}")

                    except Exception as e:
                        continue

        if not feature_model_counts:
            continue

        # Note: unlike prepare_empath_feature_stats() (SOTA side), this writes every
        # significant feature found, not just the top-N -- matching the original
        # plot_empath_feature_frequency()'s behavior, where the plot itself capped
        # its y-axis at top_n but the CSV export did not. plot_empath_feature_heatmap.py
        # re-applies its own top-N selection when reading this CSV, so the final
        # figure is unaffected either way.
        dataset_label = format_dataset_name(dataset_name)
        for feat, model_counts in feature_model_counts.items():
            for mdl, cnt in model_counts.items():
                empath_rows.append({'dataset': dataset_label, 'feature': feat, 'model': mdl, 'count': cnt})

    if not empath_rows:
        print("  No data found for Empath feature stats!")
        return

    stats_path = os.path.join(OUTPUT_DIR, f'config_empath_stats_{response_type}.csv')
    pd.DataFrame(empath_rows).to_csv(stats_path, index=False)
    print(f"  Stats saved to: {stats_path}")


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

    stats_path = os.path.join(OUTPUT_DIR, f'config_feature_importance_stats.csv')
    df.groupby(['dataset', 'model', 'feature'])['importance'].mean().reset_index().rename(
        columns={'importance': 'mean_importance'}
    ).to_csv(stats_path, index=False)
    print(f"  Stats saved to: {stats_path}")

    with with_plot_style(PRESENTATION_MODE):
        dataset_order = NORMALIZED_DATASETS
        datasets = [d for d in dataset_order if d in df['dataset'].unique()]

        models = get_ordered_models(df['model'].unique())

        overall_importance = df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        top_overall = set(overall_importance.head(max_features).index)

        top_per_model = set()
        for model in models:
            model_data = df[df['model'] == model]
            model_importance = model_data.groupby('feature')['importance'].mean().sort_values(ascending=False)
            top_per_model.update(model_importance.head(2).index)

        candidate_features = top_overall.union(top_per_model)

        print(f"  Candidate features: {len(candidate_features)}")

        fig, axes = plt.subplots(1, 3, figsize=(22, 5), sharey=True)

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

        cmap = 'Blues'

        from matplotlib.colors import PowerNorm

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

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]

            dataset_data = df[df['dataset'] == dataset]

            dataset_importance = dataset_data.groupby('feature')['importance'].mean().sort_values(ascending=False)

            dataset_top = set(dataset_importance.head(max_features).index)

            dataset_top_per_model = set()
            for model in models:
                model_data = dataset_data[dataset_data['model'] == model]
                if not model_data.empty:
                    model_importance = model_data.groupby('feature')['importance'].mean().sort_values(ascending=False)
                    dataset_top_per_model.update(model_importance.head(2).index)

            selected_features_for_dataset = dataset_top.union(dataset_top_per_model).intersection(candidate_features)

            selected_features = sorted(
                selected_features_for_dataset,
                key=lambda x: dataset_importance.get(x, 0),
                reverse=True
            )[:max_features]

            print(f"  {dataset} - Selected {len(selected_features)} features")

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

            if use_shift:
                importance_array_shifted = importance_array + shift
            else:
                importance_array_shifted = importance_array

            im = ax.imshow(importance_array_shifted, cmap=cmap, aspect='auto', norm=norm)

            formatted_features = [format_feature_name(f) for f in selected_features]

            ax.set_xticks(range(len(formatted_features)))
            ax.set_xticklabels(formatted_features, rotation=45, ha='right', fontsize=14)

            if idx == 0:
                ax.set_yticks(range(len(models)))
                ax.set_yticklabels([])

                for i, model in enumerate(models):
                    ax.text(-0.5, i, model, ha='right', va='center', color='black',
                           fontsize=14)

            font_size = 9
            for i in range(len(models)):
                for j in range(len(formatted_features)):
                    value = importance_array[i, j]
                    value_shifted = importance_array_shifted[i, j]
                    normalized_value = (value_shifted - vmin_shifted) / (vmax_shifted - vmin_shifted)
                    text_color = 'white' if normalized_value > 0.5 else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                           color=text_color, fontsize=font_size)

            dataset_label = format_dataset_name(dataset)
            dataset_color = get_dataset_color(dataset)
            ax.set_title(dataset_label, fontsize=15, pad=10, color=dataset_color)

            if idx > 0:
                ax.set_yticks([])

        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Feature Importance', fontsize=14)

        output_path = os.path.join(OUTPUT_DIR, f'feature_importance_heatmap_top{max_features}.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close()

        print(f"  Saved to: {output_path}")


def main():
    global DATASETS, NORMALIZED_DATASETS, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Generate main-text configuration-optimization figures.")
    parser.add_argument("results_folder", nargs="?", default=None,
                        help="Base results folder (e.g. results_2026_03_17). "
                             "Platform subfolders (bluesky, reddit, twitter) are detected automatically.")
    args = parser.parse_args()

    if args.results_folder is not None:
        base = Path(args.results_folder)
        EXCLUDE_DIRS = {'cosine_baselines', 'SOTA_figures', 'configuration_optimization_figures', 'figures'}
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
    print("Configuration Optimization: Main-Text Figures")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Datasets: {DATASETS}")

    print("\n=== Loading Accuracy Data ===")
    df = load_configuration_data(DATASETS)

    if df.empty:
        print("No accuracy data found!")
        return

    print(f"Loaded {len(df)} records")

    df_best_configs = save_best_configurations(df, OUTPUT_DIR, response_type="random", metric="accuracy")

    if df_best_configs is None:
        print("Could not determine best configurations!")
        return

    plot_sota_vs_best_performance(df, response_type="random", metric="accuracy")
    plot_intervention_steps_aggregated(df, response_type="random", metric="accuracy")
    plot_configuration_consistency_with_baseline(df, metric="accuracy")

    similarity_data = load_cosine_similarities_by_response_type(DATASETS)

    if similarity_data['random'] is not None and not similarity_data['random'].empty:
        plot_cosine_similarity_boxplot(similarity_data, df_best_configs)
        plot_cosine_similarity_boxplot_all_methods(similarity_data, df_best_configs)

    prepare_config_empath_feature_stats(DATASETS, df_best_configs)

    df_importance = load_feature_importance_data(DATASETS, df_best_configs, response_type='random')
    if df_importance is not None:
        plot_feature_importance_heatmap(df_importance, max_features=10)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output files in: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
