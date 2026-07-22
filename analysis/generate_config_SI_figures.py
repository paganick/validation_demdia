#!/usr/bin/env python3
"""
Configuration-optimization ("best config" per model/dataset) figures that
appear only in the Supplementary Information:
  - Response overlap summary across selection methods
  - Cosine similarity by model, split by selection method
  - Raw feature bias (best-config AI vs human distributions)

For the main-text configuration-optimization figures (SOTA vs best, stepwise
intervention, cosine similarity boxplots, feature importance heatmap, Empath
frequency, configuration consistency), see generate_config_main_figures.py.

Usage:
    python analysis/generate_config_SI_figures.py results_cleaned/
"""

import os
import sys
import argparse
import glob
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
    _save_stats, HUMAN_COLOR, HUMAN_LABEL,
    load_configuration_data, load_cosine_similarities_by_response_type,
    save_best_configurations, load_feature_importance_data,
)

# === Configuration ===
DATASETS = ['results_cleaned_20260309_095640/bluesky']
OUTPUT_DIR = "results_revision/configuration_optimization_figures"
PRESENTATION_MODE = False
SAVE_FORMAT = 'png'
NORMALIZED_DATASETS = []  # populated in main()


# === Response Overlap Summary ===
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
    print(f"\n=== Generating Figure: Response Overlap Summary ===")

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
    stats_path = os.path.join(OUTPUT_DIR, f'config_overlap_stats.csv')
    df_summary.to_csv(stats_path, index=False)
    print(f"  Stats saved to: {stats_path}")
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
            tick.set_color('black')
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


# === Cosine Similarity by Model x Config Type ===
def plot_cosine_similarity_by_model_all_methods(similarity_data, df_best_configs):
    """
    3 rows (one per platform) x model groups on x-axis.
    Within each model group: 4 boxplots for the 4 config types
    (Baseline, Best Random, Best Cosine Optimal, Best ML Optimal),
    colored by config type.

    Analogous to SOTA_cosine_similarity_by_model.png but split into
    the 4 configurations from sota_vs_best_cosine_similarity_boxplot_all_methods.png.
    """
    print(f"\n=== Generating Figure: Cosine Similarity by Model x Config Type ===")

    df_random = similarity_data['random']
    df_ml = similarity_data['ML_optimal']
    df_cosine = similarity_data['cosine_optimal']

    if df_random.empty or df_ml.empty or df_cosine.empty:
        print("  Missing data for some response types")
        return

    df_baseline = df_random[
        (df_random['persona'] == 1) &
        (df_random['style'] == 0) &
        (df_random['context'] == 0) &
        (df_random['ft'] == 0)
    ].copy()
    df_baseline['type'] = 'Baseline (BL+PE)'

    response_datasets = [
        (df_random, 'Best Random'),
        (df_cosine, 'Best Cosine Optimal'),
        (df_ml, 'Best ML Optimal'),
    ]

    all_best_data = []
    for df_source, type_label in response_datasets:
        pieces = []
        for _, cfg in df_best_configs.iterrows():
            chunk = df_source[
                (df_source['model'] == cfg['model']) &
                (df_source['dataset'] == cfg['dataset']) &
                (df_source['persona'] == int(cfg['has_persona'])) &
                (df_source['style'] == int(cfg['has_style'])) &
                (df_source['context'] == int(cfg['has_context'])) &
                (df_source['ft'] == int(cfg['has_finetuning']))
            ].copy()
            if len(chunk) > 0:
                chunk['type'] = type_label
                pieces.append(chunk)
        if pieces:
            all_best_data.append(pd.concat(pieces, ignore_index=True))

    if not all_best_data:
        print("  No best-config data found")
        return

    df_combined = pd.concat([df_baseline] + all_best_data, ignore_index=True)

    stats_path = os.path.join(OUTPUT_DIR, 'config_cosine_by_model_all_methods_stats.csv')
    _save_stats(df_combined, ['dataset', 'model', 'type'], 'similarity', stats_path)

    config_types = [
        'Baseline (BL+PE)',
        'Best Random',
        'Best Cosine Optimal',
        'Best ML Optimal',
    ]
    type_colors = {
        'Baseline (BL+PE)':      '#999999',
        'Best Random':           '#4C72B0',
        'Best Cosine Optimal':   '#DD8452',
        'Best ML Optimal':       '#55A868',
    }

    datasets    = NORMALIZED_DATASETS
    model_order = get_ordered_models(df_combined['model'].unique())
    n_types     = len(config_types)
    box_width   = 0.16
    group_span  = n_types * box_width
    group_gap   = 0.35
    group_step  = group_span + group_gap

    def group_positions(model_idx):
        centre = model_idx * group_step
        offsets = [(i - (n_types - 1) / 2) * box_width for i in range(n_types)]
        return [centre + o for o in offsets]

    with with_plot_style(PRESENTATION_MODE):
        fig, axes = plt.subplots(
            len(datasets), 1,
            figsize=(max(14, 2.2 * len(model_order)), 5 * len(datasets)),
            sharex=True,
        )
        if len(datasets) == 1:
            axes = [axes]

        for row_idx, (ax, dataset) in enumerate(zip(axes, datasets)):
            all_pos, all_data, all_colors = [], [], []

            for m_idx, model in enumerate(model_order):
                positions = group_positions(m_idx)
                for t_idx, ctype in enumerate(config_types):
                    vals = df_combined[
                        (df_combined['dataset'] == dataset) &
                        (df_combined['model']   == model) &
                        (df_combined['type']    == ctype)
                    ]['similarity'].values
                    all_pos.append(positions[t_idx])
                    all_data.append(vals)
                    all_colors.append(type_colors[ctype])

            bp = ax.boxplot(
                all_data,
                positions=all_pos,
                widths=box_width * 0.85,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markersize=2, alpha=0.35),
            )
            for patch, color in zip(bp['boxes'], all_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor('black')
                patch.set_linewidth(0.8)
            for i, color in enumerate(all_colors):
                bp['whiskers'][i * 2].set_color(color)
                bp['whiskers'][i * 2 + 1].set_color(color)
                bp['caps'][i * 2].set_color(color)
                bp['caps'][i * 2 + 1].set_color(color)

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_ylim([-1, 1])
            ax.set_ylabel("Cosine Similarity", fontsize=13)
            ax.tick_params(axis='y', labelsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)

            dataset_color = get_dataset_color(dataset)
            ax.set_title(format_dataset_name(dataset), fontsize=14, loc='right',
                         color=dataset_color, pad=6)

            centres = [m_idx * group_step for m_idx in range(len(model_order))]
            ax.set_xticks(centres)
            if row_idx == len(datasets) - 1:
                ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=12)
                for tick, model in zip(ax.get_xticklabels(), model_order):
                    tick.set_color('black')
            else:
                ax.set_xticklabels([])

            for m_idx in range(len(model_order) - 1):
                sep_x = (m_idx * group_step + group_span / 2) + group_gap / 2
                ax.axvline(sep_x, color='lightgray', linewidth=0.8, linestyle='-', zorder=0)

        legend_handles = [
            Patch(facecolor=type_colors[ct], alpha=0.75, edgecolor='black', label=ct)
            for ct in config_types
        ]
        fig.legend(handles=legend_handles, loc='lower center',
                   bbox_to_anchor=(0.5, -0.12), ncol=len(config_types),
                   fontsize=12, frameon=True, title='Configuration type', title_fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.10)
        output_path = os.path.join(OUTPUT_DIR, 'config_cosine_by_model_all_methods.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")


# === Raw Feature Bias (best-config, AI vs Human) ===
def generate_feature_bias_best_config(df_best_configs, top_n: int = 10, df_importance=None):
    """
    3 rows (platforms) x top_n columns (features).
    Each subplot: one boxplot per model (MODEL_PALETTE) + one Human boxplot (gray),
    plus a dashed horizontal line at the human median.
    Uses best configuration per model/dataset instead of baseline+persona-only files.

    When df_importance is supplied the same global candidate_features strategy used
    by the heatmap is applied, keeping the two plots in sync.
    """
    print(f"\n=== Generating Figure: Raw Feature Bias (best-config, AI vs Human) ===")

    if df_importance is not None and not df_importance.empty:
        _overall = df_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
        _top_overall = set(_overall.head(10).index)
        _top_per_model = set()
        for _m in df_importance['model'].unique():
            _mi = (df_importance[df_importance['model'] == _m]
                   .groupby('feature')['importance'].mean()
                   .sort_values(ascending=False))
            _top_per_model.update(_mi.head(2).index)
        candidate_features = _top_overall.union(_top_per_model)
        all_models = df_importance['model'].unique().tolist()
    else:
        candidate_features = None
        all_models = None

    platform_features: dict = {}
    platform_data: dict = {}

    for dataset in DATASETS:
        pname = normalize_dataset_name(dataset)
        pdata_configs = df_best_configs[df_best_configs['dataset'] == pname]
        if pdata_configs.empty:
            print(f"  Skipping {pname}: no best configs found")
            continue

        feat_importance_dfs = []
        for _, cfg_row in pdata_configs.iterrows():
            model = cfg_row['model']
            expected = {
                'ft':      int(cfg_row['has_finetuning']),
                'context': int(cfg_row['has_context']),
                'style':   int(cfg_row['has_style']),
                'persona': int(cfg_row['has_persona']),
            }
            pattern = str(Path(dataset) / '**' / '*feature_correlation_stats.csv')
            for fpath in glob.glob(pattern, recursive=True):
                if 'random_validation' not in fpath:
                    continue
                try:
                    fmodel, fft, fctx, fstyle, fpersona = parse_filename(fpath)
                    if (fmodel == model
                            and fft      == expected['ft']
                            and fctx     == expected['context']
                            and fstyle   == expected['style']
                            and fpersona == expected['persona']):
                        df_imp = pd.read_csv(fpath)
                        if 'feature' in df_imp.columns and 'importance' in df_imp.columns:
                            df_imp = df_imp[['feature', 'importance']].copy()
                            df_imp['model'] = model
                            feat_importance_dfs.append(df_imp)
                except Exception:
                    continue

        if not feat_importance_dfs:
            print(f"  Skipping {pname}: no feature importance files matched best config")
            continue

        combined_imp = pd.concat(feat_importance_dfs, ignore_index=True)
        di = combined_imp.groupby('feature')['importance'].mean().sort_values(ascending=False)

        if candidate_features is not None:
            dataset_top = set(di.head(10).index)
            dataset_top_per_model = set()
            for _m in (all_models or []):
                _mi = (combined_imp[combined_imp['model'] == _m]
                       .groupby('feature')['importance'].mean()
                       .sort_values(ascending=False))
                dataset_top_per_model.update(_mi.head(2).index)
            top_feats = sorted(
                dataset_top.union(dataset_top_per_model).intersection(candidate_features),
                key=lambda x: float(di.get(x, 0)),
                reverse=True
            )[:top_n]
        else:
            top_feats = di.head(top_n).index.tolist()

        print(f"  {pname}: top features = {top_feats}")

        model_data = {}
        for _, cfg_row in pdata_configs.iterrows():
            model = cfg_row['model']
            expected = {
                'ft':      int(cfg_row['has_finetuning']),
                'context': int(cfg_row['has_context']),
                'style':   int(cfg_row['has_style']),
                'persona': int(cfg_row['has_persona']),
            }
            pattern = str(Path(dataset) / '**' / '*_random_validation_data.csv')
            for lpath in glob.glob(pattern, recursive=True):
                if '_features' in lpath:
                    continue
                try:
                    fmodel, fft, fctx, fstyle, fpersona = parse_filename(lpath)
                    if (fmodel == model
                            and fft      == expected['ft']
                            and fctx     == expected['context']
                            and fstyle   == expected['style']
                            and fpersona == expected['persona']):
                        feat_path = lpath.replace(
                            '_random_validation_data.csv',
                            '_random_validation_data_features.csv'
                        )
                        if not os.path.exists(feat_path):
                            continue
                        labels_df = pd.read_csv(lpath)
                        feats_df  = pd.read_csv(feat_path)
                        if len(labels_df) != len(feats_df):
                            continue
                        feats_df['label'] = labels_df['labels'].values
                        model_data[model] = feats_df
                        break
                except Exception:
                    continue

        if model_data:
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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3.5 * n_rows), sharey=False)
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
                    patch.set_facecolor(color); patch.set_alpha(1.0)
                for i, color in enumerate(box_colors):
                    bp['whiskers'][i*2].set_color(color);   bp['whiskers'][i*2+1].set_color(color)
                    bp['caps'][i*2].set_color(color);       bp['caps'][i*2+1].set_color(color)

                rng = np.random.default_rng(42)
                max_pts = 80
                for pos, vals, color in zip(x_positions, box_data, box_colors):
                    if len(vals) == 0:
                        continue
                    sample = vals if len(vals) <= max_pts else rng.choice(vals, max_pts, replace=False)
                    jitter = rng.uniform(-0.18, 0.18, size=len(sample))
                    ax.scatter(pos + jitter, sample, s=4, color=color, alpha=0.45, zorder=3, linewidths=0)

                all_vals = np.concatenate([v for v in box_data if len(v) > 0])
                if len(all_vals) > 0:
                    lo, hi = np.percentile(all_vals, 2), np.percentile(all_vals, 98)
                    pad = (hi - lo) * 0.1 if hi > lo else 0.1
                    ax.set_ylim(lo - pad, hi + pad)

                ax.axhline(human_median, color=HUMAN_COLOR, linestyle='-', linewidth=2.5, alpha=1.0, zorder=4)
                ax.set_title(feat.replace('_', ' '), fontsize=16, pad=4)
                if col_idx == 0:
                    ax.set_ylabel(format_dataset_name(pname), fontsize=18, color=pcolor, labelpad=6)
                ax.tick_params(axis='y', labelsize=9)
                ax.tick_params(axis='x', bottom=False)
                ax.set_xticks([])
                ax.grid(True, alpha=0.3, axis='y')

        legend_handles = [Patch(facecolor=MODEL_PALETTE.get(m, '#888888'), alpha=1.0, label=m)
                          for m in model_order]
        legend_handles.append(Patch(facecolor=HUMAN_COLOR, alpha=1.0, label=HUMAN_LABEL))
        legend = fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.01),
                   ncol=min(len(legend_handles), 5), fontsize=16, frameon=True)
        for text, model in zip(legend.get_texts(), model_order):
            text.set_color('black')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        output_path = os.path.join(OUTPUT_DIR, f'config_feature_bias.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")


def main():
    global DATASETS, NORMALIZED_DATASETS, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Generate SI-only configuration-optimization figures.")
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
        OUTPUT_DIR = str(base / "figures" / "SI_figures")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Configuration Optimization: SI-Only Figures")
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

    df_overlap = analyze_response_overlap(DATASETS)
    if df_overlap is not None:
        plot_response_overlap_summary(df_overlap)

    similarity_data = load_cosine_similarities_by_response_type(DATASETS)

    if similarity_data['random'] is not None and not similarity_data['random'].empty:
        plot_cosine_similarity_by_model_all_methods(similarity_data, df_best_configs)

    df_importance = load_feature_importance_data(DATASETS, df_best_configs, response_type='random')
    generate_feature_bias_best_config(df_best_configs, top_n=10, df_importance=df_importance)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output files in: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
